"""Main orchestrator: process all PDFs in test_data, parse pages, write CSVs."""

import asyncio
import csv
import io
import logging
import os
import re
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import PyPDF2
from dotenv import load_dotenv

# Load .env before importing gemini_client, so module-level config
# (e.g., MAX_CONCURRENT) can read environment overrides.
load_dotenv()

from gemini_client import (
    CoverMetadata,
    MAX_CONCURRENT,
    PageData,
    SAVE_RAW_PAGE_LOGS,
    create_client,
    extract_cover_metadata,
    extract_page_async,
)

# --- Paths ---
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "data_list_pdfs"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output_list_pdfs"))
LOG_ROOT_DIR = Path(os.environ.get("LOG_ROOT_DIR", "logs"))

# Pages to process (0-indexed). Skip page 0 (cover) and page 1 (maps/photos).
FIRST_DATA_PAGE = 2

CSV_COLUMNS = [
    "serial_number",
    "epic_number",
    "name",
    "relation_type",
    "relation_name",
    "house_number",
    "age",
    "gender",
    "section_raw",
    "part",
    "ward",
    "corporation",
    "pincode",
    "pollingname",
    "pollingaddress",
]


def _env_int(name: str, default: int) -> int:
    """Read positive integer env var with safe fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logging.warning("Invalid %s=%r; using default=%s", name, raw, default)
        return default
    if value <= 0:
        logging.warning("Non-positive %s=%r; using default=%s", name, raw, default)
        return default
    return value


def load_api_keys() -> list[str]:
    """Load Gemini API keys from GEMINI_API_KEYS or GEMINI_API_KEY."""
    keys: list[str] = []
    raw_multi = os.environ.get("GEMINI_API_KEYS", "")
    if raw_multi:
        for token in raw_multi.replace("\n", ",").split(","):
            key = token.strip()
            if key:
                keys.append(key)

    if not keys:
        single = os.environ.get("GEMINI_API_KEY", "").strip()
        if single:
            keys.append(single)

    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)

    if not deduped:
        raise KeyError(
            "No Gemini API keys configured. Set GEMINI_API_KEY or GEMINI_API_KEYS."
        )
    return deduped


def normalize_gender(raw: str) -> str:
    """Normalize gender values to 'M' or 'F'."""
    val = raw.strip().upper()
    if val in ("M", "MALE"):
        return "M"
    if val in ("F", "FEMALE"):
        return "F"
    return raw


def extract_single_page_pdf(reader: PyPDF2.PdfReader, page_index: int) -> bytes:
    """Extract one page from a PdfReader as a standalone PDF byte buffer."""
    writer = PyPDF2.PdfWriter()
    writer.add_page(reader.pages[page_index])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def page_data_path(page_data_dir: Path, page_number: int) -> Path:
    """Return canonical parsed JSON path for one page (1-indexed)."""
    return page_data_dir / f"page_{page_number:03d}.json"


def write_page_data(page_data_dir: Path, page_number: int, page_data: PageData) -> None:
    """Atomically persist canonical parsed output for one page."""
    out_path = page_data_path(page_data_dir, page_number)
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(page_data.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(out_path)


def read_page_data(page_data_dir: Path, page_number: int) -> PageData | None:
    """Read one canonical page output. Invalid files are treated as missing."""
    in_path = page_data_path(page_data_dir, page_number)
    if not in_path.exists():
        return None
    try:
        return PageData.model_validate_json(in_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning(
            f"Invalid cached page file {in_path}: {exc}. "
            "Page will be reprocessed."
        )
        return None


def discover_completed_pages(
    page_data_dir: Path, page_indices: Iterable[int]
) -> set[int]:
    """Find completed pages from canonical per-page outputs."""
    completed: set[int] = set()
    for page_idx in page_indices:
        page_num = page_idx + 1
        if read_page_data(page_data_dir, page_num) is not None:
            completed.add(page_idx)
    return completed


def load_cover_metadata(cover_metadata_path: Path) -> CoverMetadata | None:
    """Load cached cover metadata from disk."""
    if not cover_metadata_path.exists():
        return None
    try:
        return CoverMetadata.model_validate_json(
            cover_metadata_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        logging.warning(
            f"Invalid cover metadata cache ({cover_metadata_path.name}): {exc}. "
            "Will re-extract from cover page."
        )
        return None


def save_cover_metadata(cover_metadata_path: Path, meta: CoverMetadata) -> None:
    """Persist cover metadata atomically."""
    tmp_path = cover_metadata_path.with_suffix(".json.tmp")
    tmp_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(cover_metadata_path)


def get_cover_metadata(
    clients,
    reader: PyPDF2.PdfReader,
    cover_metadata_path: Path,
    cover_key_seed: int,
) -> CoverMetadata:
    """Get cover metadata from cache or extract it from page 1."""
    cached = load_cover_metadata(cover_metadata_path)
    if cached is not None:
        return cached

    cover_pdf_bytes = extract_single_page_pdf(reader, 0)
    extracted = extract_cover_metadata(
        clients, cover_pdf_bytes, base_index=cover_key_seed
    )
    save_cover_metadata(cover_metadata_path, extracted)
    return extracted


def rebuild_csv_from_pages(
    page_data_dir: Path,
    page_indices: list[int],
    csv_path: Path,
    cover_meta: CoverMetadata,
) -> None:
    """Rebuild CSV atomically from all available canonical page outputs."""
    tmp_path = csv_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for page_idx in sorted(page_indices):
            page_num = page_idx + 1
            page_data = read_page_data(page_data_dir, page_num)
            if page_data is None:
                continue
            for v in page_data.voters:
                writer.writerow(
                    {
                        "serial_number": v.serial_number,
                        "epic_number": v.voter_id_epic,
                        "name": v.name_english,
                        "relation_type": v.relationship_type,
                        "relation_name": v.relative_name_english,
                        "house_number": v.house_number,
                        "age": v.age,
                        "gender": normalize_gender(v.gender),
                        "section_raw": page_data.section_raw,
                        "part": cover_meta.part,
                        "ward": cover_meta.ward,
                        "corporation": cover_meta.corporation,
                        "pincode": cover_meta.pincode,
                        "pollingname": cover_meta.pollingname,
                        "pollingaddress": cover_meta.pollingaddress,
                    }
                )
    tmp_path.replace(csv_path)


def validate_results(csv_path: Path) -> None:
    """Validate basic output consistency for one extracted CSV."""
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    male = sum(1 for r in rows if r["gender"] == "M")
    female = sum(1 for r in rows if r["gender"] == "F")

    print("\n--- Validation ---")
    print(f"CSV:           {csv_path.name}")
    print(f"Total records: {total}")
    print(f"Male:          {male}")
    print(f"Female:        {female}")

    serials = [int(r["serial_number"]) for r in rows]
    dupes = {s for s, count in Counter(serials).items() if count > 1}
    if dupes:
        print(f"WARNING: Duplicate serial numbers found: {sorted(dupes)}")


async def process_pages(
    clients,
    reader: PyPDF2.PdfReader,
    page_indices: list[int],
    completed: set[int],
    cover_meta: CoverMetadata,
    page_data_dir: Path,
    csv_path: Path,
    log_dir: Path | None,
    pdf_name: str,
    page_concurrency: int,
) -> None:
    """Process pages with a rolling worker queue and end-of-run CSV rebuild."""
    semaphore = asyncio.Semaphore(page_concurrency)
    max_queue_attempts_per_page = 2

    pending: deque[int] = deque(idx for idx in page_indices if idx not in completed)
    wrote_new_pages = False
    if not pending:
        logging.info("[%s] All pages already completed.", pdf_name)
        if not csv_path.exists():
            rebuild_csv_from_pages(page_data_dir, page_indices, csv_path, cover_meta)
            logging.info("[%s] Built missing CSV from cached pages.", pdf_name)
        return

    logging.info(
        "[%s] Processing %s pages with rolling concurrency=%s",
        pdf_name,
        len(pending),
        page_concurrency,
    )
    page_attempts: dict[int, int] = {}
    failed_exhausted: list[int] = []
    in_flight: dict[asyncio.Task, int] = {}
    page_bytes_cache: dict[int, bytes] = {}

    while pending or in_flight:
        while pending and len(in_flight) < page_concurrency:
            page_idx = pending.popleft()
            page_num = page_idx + 1
            if page_idx not in page_bytes_cache:
                page_bytes_cache[page_idx] = extract_single_page_pdf(reader, page_idx)
            task = asyncio.create_task(
                extract_page_async(
                    clients,
                    page_bytes_cache[page_idx],
                    page_num,
                    log_dir,
                    semaphore,
                )
            )
            in_flight[task] = page_idx

        if in_flight:
            in_flight_pages = sorted(page_idx + 1 for page_idx in in_flight.values())
            logging.info(
                "[%s] In-flight pages for %s (%s): %s",
                pdf_name,
                pdf_name,
                len(in_flight_pages),
                in_flight_pages,
            )

        done, _ = await asyncio.wait(
            in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        successful_results: list[tuple[int, PageData]] = []
        for task in done:
            page_idx = in_flight.pop(task)
            page_num = page_idx + 1
            try:
                result_page_num, page_data = task.result()
                successful_results.append((result_page_num, page_data))
            except Exception as result:
                page_attempts[page_idx] = page_attempts.get(page_idx, 0) + 1
                attempt_no = page_attempts[page_idx]
                logging.error(
                    "  Page %s failed after extractor retries "
                    "(queue-attempt %s/%s): %s",
                    page_num,
                    attempt_no,
                    max_queue_attempts_per_page,
                    result,
                )
                if attempt_no < max_queue_attempts_per_page:
                    pending.append(page_idx)
                    logging.info("  Re-queued page %s for retry.", page_num)
                else:
                    failed_exhausted.append(page_idx)

        successful_results.sort(key=lambda item: item[0])
        for page_num, page_data in successful_results:
            logging.info(
                f"  Page {page_num}: section='{page_data.section_raw[:60]}', "
                f"voters={len(page_data.voters)}"
            )
            write_page_data(page_data_dir, page_num, page_data)
            completed.add(page_num - 1)
            wrote_new_pages = True

        if successful_results:
            logging.info(
                "[%s] Progress: %s/%s pages cached for this PDF",
                pdf_name,
                len(completed),
                len(page_indices),
            )

    if failed_exhausted:
        failed_page_nums = sorted({idx + 1 for idx in failed_exhausted})
        raise RuntimeError(
            "Failed pages after repeated batch attempts: "
            f"{failed_page_nums}. Re-run to retry remaining pages."
        )

    # Build/update per-PDF CSV once per PDF run (not per progress chunk).
    if wrote_new_pages or not csv_path.exists():
        rebuild_csv_from_pages(page_data_dir, page_indices, csv_path, cover_meta)
        logging.info("[%s] Rebuilt CSV after page processing completed.", pdf_name)


def discover_pdf_inputs(input_dir: Path) -> list[Path]:
    """Return all PDFs under input_dir in deterministic order."""
    return sorted(input_dir.rglob("*.pdf"))


def ward_key_for_pdf(pdf_path: Path, input_dir: Path) -> str:
    """Resolve ward folder key for one PDF, e.g. '5_40'."""
    rel_parent = pdf_path.parent.relative_to(input_dir)
    if str(rel_parent) not in ("", "."):
        return str(rel_parent)

    match = re.match(r"^(\d+)_(\d+)_\d+_EPUB$", pdf_path.stem)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return "unknown_ward"


def process_one_pdf(
    api_keys: tuple[str, ...],
    pdf_path: Path,
    page_concurrency: int,
    cover_key_seed: int,
) -> None:
    """Run extraction pipeline for one PDF file."""
    print(f"\n=== START PDF: {pdf_path.name} ===", flush=True)
    logging.info(f"Starting PDF: {pdf_path.name}")
    clients = [create_client(api_key) for api_key in api_keys]

    pdf_key = pdf_path.stem
    ward_key = ward_key_for_pdf(pdf_path, INPUT_DIR)
    ward_output_dir = OUTPUT_DIR / ward_key
    csv_path = ward_output_dir / f"{pdf_key}.csv"
    doc_output_dir = ward_output_dir / pdf_key
    page_data_dir = doc_output_dir / "pages"
    cover_metadata_path = doc_output_dir / "cover_metadata.json"
    log_dir = LOG_ROOT_DIR / ward_key / pdf_key if SAVE_RAW_PAGE_LOGS else None

    ward_output_dir.mkdir(parents=True, exist_ok=True)
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    page_data_dir.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    reader = PyPDF2.PdfReader(str(pdf_path))
    logging.info(f"[{pdf_path.name}] Loaded PDF with {len(reader.pages)} pages")

    cover_meta = get_cover_metadata(
        clients, reader, cover_metadata_path, cover_key_seed
    )
    logging.info(
        (
            "[%s] Cover metadata: part='%s', ward='%s', corporation='%s', "
            "pincode='%s', pollingname='%s', start='%s', end='%s', "
            "male='%s', female='%s', third_gender='%s', total='%s'"
        ),
        pdf_path.name,
        cover_meta.part,
        cover_meta.ward,
        cover_meta.corporation,
        cover_meta.pincode,
        cover_meta.pollingname,
        cover_meta.starting_serial_number,
        cover_meta.ending_serial_number,
        cover_meta.male,
        cover_meta.female,
        cover_meta.third_gender,
        cover_meta.total,
    )

    if len(reader.pages) <= FIRST_DATA_PAGE:
        logging.warning(
            f"[{pdf_path.name}] No data pages found (page count={len(reader.pages)})."
        )
        return

    page_indices = list(range(FIRST_DATA_PAGE, len(reader.pages)))
    completed = discover_completed_pages(page_data_dir, page_indices)
    if completed:
        logging.info(
            f"[{pdf_path.name}] Resuming from page cache. "
            f"Already completed {len(completed)} pages."
        )

    asyncio.run(
        process_pages(
            clients,
            reader,
            page_indices,
            completed,
            cover_meta,
            page_data_dir,
            csv_path,
            log_dir,
            pdf_path.name,
            page_concurrency,
        )
    )

    if csv_path.exists():
        validate_results(csv_path)


def main() -> None:
    api_keys = load_api_keys()

    OUTPUT_DIR.mkdir(exist_ok=True)
    if SAVE_RAW_PAGE_LOGS:
        LOG_ROOT_DIR.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Using %s Gemini API key(s) for rotation", len(api_keys))

    pdf_paths = discover_pdf_inputs(INPUT_DIR)
    if not pdf_paths:
        logging.warning(f"No PDFs found in {INPUT_DIR}")
        return

    logging.info(f"Found {len(pdf_paths)} PDF(s) in {INPUT_DIR}")
    pdf_prefetch = min(_env_int("PDF_PREFETCH", 1), len(pdf_paths))
    page_concurrency = max(1, MAX_CONCURRENT // pdf_prefetch)
    logging.info(
        "PDF concurrency=%s, page concurrency per active PDF=%s, global target=%s",
        pdf_prefetch,
        page_concurrency,
        MAX_CONCURRENT,
    )

    failures: list[tuple[Path, Exception]] = []
    with ThreadPoolExecutor(max_workers=pdf_prefetch) as pool:
        api_keys_tuple = tuple(api_keys)
        future_to_pdf = {
            pool.submit(
                process_one_pdf,
                api_keys_tuple,
                pdf_path,
                page_concurrency,
                idx,
            ): pdf_path
            for idx, pdf_path in enumerate(pdf_paths)
        }
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                future.result()
            except Exception as exc:
                logging.error("[%s] Failed: %s", pdf_path.name, exc)
                failures.append((pdf_path, exc))

    if failures:
        failed_names = ", ".join(p.name for p, _ in failures)
        raise RuntimeError(f"One or more PDFs failed: {failed_names}")


if __name__ == "__main__":
    main()
