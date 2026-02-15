"""Gemini API client: schemas, prompts, and extraction logic."""

import asyncio
import logging
import os
import random
import time
from pathlib import Path
from typing import Sequence

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
try:
    import httpx
except Exception:  # pragma: no cover - optional dependency at runtime
    httpx = None

# --- Model ---
MODEL_NAME = "gemini-3-flash-preview"
COVER_MODEL_NAME = "gemini-3-pro-preview"
RETRY_SLEEP_SECONDS = 0.1
RETRY_JITTER_SECONDS = 0.05
SAVE_RAW_PAGE_LOGS = False


def _env_int(name: str, default: int) -> int:
    """Read integer env var with safe fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %s", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Non-positive %s=%r; falling back to %s", name, raw, default)
        return default
    return value


# Max concurrent Gemini requests (configurable via env).
MAX_CONCURRENT = _env_int("MAX_CONCURRENT", 12)


# --- Pydantic schemas for structured output ---

class VoterRecord(BaseModel):
    serial_number: int = Field(description="Serial number of the voter")
    voter_id_epic: str = Field(description="EPIC voter ID string")
    name_english: str = Field(description="Voter full name in English")
    relationship_type: str = Field(description="Relation type (Father, Husband, Mother, Brother, Sister, Wife, Other)")
    relative_name_english: str = Field(description="Name of relative")
    house_number: str = Field(description="House number from the 'H. No.' field")
    age: int = Field(description="Age")
    gender: str = Field(description="Gender, exactly as listed (M/F/)")


class PageData(BaseModel):
    section_raw: str = Field(
        description="The section/address header at the top of the page, third line, not the first two (Corporation, Ward)"
    )
    voters: list[VoterRecord] = Field(
        description="List of all voter records on this page"
    )


class CoverMetadata(BaseModel):
    part: str = Field(description="Part No. only, e.g. '22'")
    ward: str = Field(
        description="Ward No. and Name, e.g. '40 - Mahalakshmipuram'"
    )
    corporation: str = Field(
        description="Corporation name, e.g. 'Bengaluru West City Corporation'"
    )
    pincode: str = Field(description="Pin Code")
    pollingname: str = Field(
        description="Polling Station No. and Name"
    )
    pollingaddress: str = Field(description="Polling Station Address")
    starting_serial_number: str = Field(
        default="",
        description="Starting Serial Number",
    )
    ending_serial_number: str = Field(
        default="",
        description="Ending Serial Number",
    )
    male: str = Field(default="", description="Net Voter Count, Male")
    female: str = Field(default="", description="Net Voter Count, Female")
    third_gender: str = Field(
        default="",
        description="Net Voter Count, Third Gender",
    )
    total: str = Field(default="", description="Net Voter Count, Total")


# --- Prompts ---

SYSTEM_PROMPT = """\
You are an expert data-entry operator extracting voter records from an Indian Electoral Roll PDF \
(Karnataka State Election Commission).

CRITICAL INSTRUCTIONS:
- IGNORE the light-gray watermark text "STATE ELECTION COMMISSION, KARANATAKA" that appears \
repeatedly across the entire background. It is NOT data.
- Focus ONLY on the high-contrast black printed text.
- Each page has a section header at the top (below the ward/part info line) and a grid of voter \
cards arranged in 3 columns and up to 10 rows (up to 30 voters per page).
- Each voter card contains: serial number, EPIC number (voter ID), name, relation type and \
relation name, house number (H. No.), age, and gender.
- The "Photo Available" placeholder in each card should be IGNORED.
- Extract ALL voter cards on the page. Do not skip any.
- If a field is missing/unclear for a voter, return the best readable value; use empty string only \
when absent.
- For relationship_type, normalize to exactly one of: "Father", "Husband", "Mother", "Brother", "Wife", "Sister", "Other".
- For gender, use first letter capitalized.
- house_number should be the raw string from the "H. No." field (e.g. "1/A", "13/1", "34/26, 2nd Floor").\
"""

USER_PROMPT = "Extract all voter records from this electoral roll page. Return the section header and every voter entry."
COVER_USER_PROMPT = (
    "Extract only these fields from this cover page: "
    "part, ward, corporation, pincode, pollingname, pollingaddress, "
    "starting_serial_number, ending_serial_number, male, female, third_gender, total."
)

COVER_SYSTEM_PROMPT = """\
You are extracting metadata from the cover page of a Karnataka electoral roll PDF.

Return strict JSON with:
- part: the part number only (e.g., "22")
- ward: ward number and name (e.g., "40 - Mahalakshmipuram")
- corporation: corporation name (e.g., "Bengaluru West City Corporation")
- pincode: pin code only (digits)
- pollingname: value from "Polling Station No. and Name"
- pollingaddress: value from "Polling Station Address"
- starting_serial_number: value for "Starting Serial Number"
- ending_serial_number: value for "Ending Serial Number"
- male: value for "Male"
- female: value for "Female"
- third_gender: value for "Third Gender" (or equivalent label if present)
- total: value for "Total"

Do not include labels such as "Part No." or "Ward No. and Name".
If a field is absent, return an empty string for that field.
If text is uncertain, return the closest readable value from the page.
"""


# --- Client and extraction ---

def create_client(api_key: str) -> genai.Client:
    """Create and return a Gemini API client."""
    return genai.Client(api_key=api_key)


def _pick_client(
    clients: Sequence[genai.Client], base_index: int, attempt: int
) -> genai.Client:
    """Select one client from pool using deterministic round-robin."""
    if not clients:
        raise ValueError("No Gemini clients provided.")
    return clients[(base_index + attempt) % len(clients)]


def is_retryable_error(err: Exception) -> bool:
    """Return True for transient API failures worth retrying."""
    if isinstance(err, genai_errors.APIError):
        code = err.code or 0
        return code == 429 or 500 <= code < 600
    if isinstance(err, (TimeoutError, ConnectionError)):
        return True
    if httpx is not None and isinstance(err, httpx.HTTPError):
        return True
    return False


def retry_delay_seconds(attempt: int) -> float:
    """Exponential backoff delay with existing base + jitter."""
    return (RETRY_SLEEP_SECONDS * (2 ** attempt)) + random.uniform(
        0, RETRY_JITTER_SECONDS
    )


def extract_page(
    clients: Sequence[genai.Client],
    page_pdf_bytes: bytes,
    page_number: int,
    log_dir: Path | None,
    max_retries: int = 3,
) -> PageData:
    """Send a single-page PDF to Gemini and get structured voter data back."""
    contents = [
        types.Part.from_bytes(data=page_pdf_bytes, mime_type="application/pdf"),
        USER_PROMPT,
    ]

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PageData,
        system_instruction=SYSTEM_PROMPT,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            client = _pick_client(clients, page_number, attempt)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )

            if SAVE_RAW_PAGE_LOGS and log_dir is not None:
                # Optional raw response log for debugging.
                log_path = log_dir / f"page_{page_number:03d}.json"
                log_path.write_text(response.text, encoding="utf-8")

            result = response.parsed
            if result is None:
                raise ValueError(
                    f"Page {page_number}: Gemini returned unparseable response"
                )
            return result

        except Exception as e:
            last_error = e
            if is_retryable_error(e):
                delay_s = retry_delay_seconds(attempt)
                logger.warning(
                    f"Page {page_number}: attempt {attempt + 1} failed ({e}). "
                    f"Retrying in {delay_s:.3f}s..."
                )
                time.sleep(delay_s)
            else:
                raise

    raise last_error


def extract_cover_metadata(
    clients: Sequence[genai.Client],
    cover_pdf_bytes: bytes,
    base_index: int = 0,
    max_retries: int = 3,
) -> CoverMetadata:
    """Extract cover-level metadata from page 1."""
    contents = [
        types.Part.from_bytes(data=cover_pdf_bytes, mime_type="application/pdf"),
        COVER_USER_PROMPT,
    ]

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CoverMetadata,
        system_instruction=COVER_SYSTEM_PROMPT,
    )

    def run_cover_extract(model_name: str, base_index: int) -> CoverMetadata:
        last_error = None
        for attempt in range(max_retries):
            try:
                client = _pick_client(clients, base_index, attempt)
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                result = response.parsed
                if result is None:
                    raise ValueError(
                        "Cover metadata: Gemini returned unparseable response "
                        f"(model={model_name})"
                    )
                return result
            except Exception as e:
                last_error = e
                if is_retryable_error(e):
                    delay_s = retry_delay_seconds(attempt)
                    logger.warning(
                        f"Cover metadata ({model_name}): attempt {attempt + 1} failed ({e}). "
                        f"Retrying in {delay_s:.3f}s..."
                    )
                    time.sleep(delay_s)
                else:
                    raise
        raise last_error

    try:
        return run_cover_extract(COVER_MODEL_NAME, base_index)
    except genai_errors.APIError as e:
        if e.code == 404 and COVER_MODEL_NAME != MODEL_NAME:
            logger.warning(
                "Cover model '%s' unavailable; falling back to '%s'.",
                COVER_MODEL_NAME,
                MODEL_NAME,
            )
            return run_cover_extract(MODEL_NAME, base_index + 1)
        raise


async def extract_page_async(
    clients: Sequence[genai.Client],
    page_pdf_bytes: bytes,
    page_number: int,
    log_dir: Path | None,
    semaphore: asyncio.Semaphore,
) -> tuple[int, PageData]:
    """Async wrapper around extract_page. Returns (page_number, PageData)."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, extract_page, clients, page_pdf_bytes, page_number, log_dir
        )
        return page_number, result
