"""Download electoral roll PDFs by sequential part number.

Current default target list includes corp=5, ward=40.
Structure is kept ready so corp/ward iteration can be expanded later.
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Literal

BASE_URL = (
    "https://gba.karnataka.gov.in/electionfiles/ENGLISH/"
    "{corp}_{ward}_{part}_EPUB.pdf"
)

# central = 1
# north = 2
# south = 3
# east = 4
# west = 5

# Edit this list to control which corp/ward targets are downloaded.
# Format per item: "corp_ward"

# WARD_TARGETS = [
#     "5_40",
#     "3_34",
#     "3_12",
#     "3_31",
#     "3_2",
# ]
# DONE

# Edit this list to control which corp/ward targets are downloaded.
# Format per item: "corp_ward"

# WARD_TARGETS = [
#     "5_41",
#     "1_14",
#     "1_17",
#     "3_32",
#     "3_33",
#     "3_36",
#     "3_37",
#     "3_49",
#     "3_55",
#     "3_65",
#     "3_72",
# ]

# WARD_TARGETS = [
#     "4_2",
#     "4_24",
#     "4_33",
#     "4_34",
#     "4_35",
#     "4_36",
#     "4_37",
#     "4_38",
#     "4_39",
#     "4_40",
#     "4_41",
#     "4_42",
#     "4_43",
#     "4_47",
#     "4_49",
#     "4_50",
#     "2_30"
# ]

# WARD_TARGETS = [
#     "3_1",
#     "3_2",
#     "3_3",
#     "3_4",
#     "3_5",
#     "3_6",
#     "3_7",
#     "3_8",
#     "3_9",
#     "3_10",
#     "3_11",
#     "3_12",
#     "3_13",
#     "3_14",
#     "3_15",
#     "3_16",
#     "3_17",
#     "3_18",
#     "3_19",
#     "3_20",
#     "3_21",
#     "3_22",
#     "3_23",
#     "3_24",
#     "3_25",
#     "3_26",
#     "3_27",
#     "3_28",
#     "3_29",
#     "3_30",
#     "3_35"
# ]

# WARD_TARGETS = [
#     "3_35",
#     "3_37",
#     "3_38",
#     "3_39",
#     "3_40",
#     "3_41",
#     "3_42",
#     "3_43",
#     "3_44",
#     "3_45",
#     "3_46",
#     "3_47",
#     "3_48",
#     "3_50",
#     "3_51",
#     "3_52",
#     "3_53",
#     "3_54",
#     "3_56",
#     "3_57",
#     "3_58",
#     "3_59",
#     "3_60",
#     "3_61",
#     "3_62",
#     "3_63",
#     "3_64",
#     "3_66",
#     "3_67",
#     "3_68",
#     "3_69",
#     "3_70",
#     "3_71",
# ]

WARD_TARGETS = [
        "5_24",
    "5_25",
    "5_40",
    "5_41",
    "5_108",
    "5_109",
    "5_110",
    "5_111",
    "5_112"
]




def download_pdf_with_curl(url: str, dest: Path, timeout: int = 120) -> Literal["ok", "miss"]:
    """Download via curl and infer result from HTTP status."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    cmd = [
        "curl",
        "-L",
        "-sS",
        "--max-time",
        str(timeout),
        "-o",
        str(tmp),
        "-w",
        "%{http_code}",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"curl failed (exit={proc.returncode}): {proc.stderr.strip()}"
        )

    status = proc.stdout.strip()
    if status == "200":
        tmp.replace(dest)
        print(f"saved: {dest.name}")
        return "ok"

    if tmp.exists():
        tmp.unlink()

    if status == "404":
        return "miss"

    raise RuntimeError(f"unexpected HTTP status {status} for {url}")


def download_pdf(url: str, dest: Path, timeout: int = 120) -> Literal["ok", "miss"]:
    """Download one PDF.

    Returns:
    - "ok": file exists locally or was downloaded successfully
    - "miss": URL returned 404 (missing part)
    """
    if dest.exists():
        print(f"skip: {dest.name} (already exists)")
        return "ok"

    # Prefer curl for this host due Python SSL verification issues in some envs.
    if shutil.which("curl"):
        return download_pdf_with_curl(url, dest, timeout=timeout)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"unexpected HTTP status {resp.status} for {url}")
            data = resp.read()
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return "miss"
        raise

    tmp.write_bytes(data)
    tmp.replace(dest)
    print(f"saved: {dest.name}")
    return "ok"


def download_with_retries(
    url: str, dest: Path, retries: int = 3
) -> tuple[Literal["ok", "miss"], str]:
    """Retry transient failures.

    Returns:
    - ("ok", "")
    - ("miss", reason)
    """
    for attempt in range(1, retries + 1):
        try:
            status = download_pdf(url, dest)
            if status == "ok":
                return "ok", ""
            return "miss", "404 not found"
        except (urllib.error.URLError, RuntimeError, TimeoutError) as err:
            if attempt == retries:
                return "miss", f"request failed after retries: {err}"
            wait_s = 2 ** (attempt - 1)
            print(f"retry {attempt}/{retries}: {dest.name} in {wait_s}s ({err})")
            time.sleep(wait_s)

    return "miss", "unknown error"


def iter_targets(corp: int | None, ward: int | None) -> list[tuple[int, int]]:
    """Return target corp/ward pairs.

    Kept separate so it can later expand to corp/ward ranges.
    """
    if corp is not None and ward is not None:
        return [(corp, ward)]
    if corp is not None or ward is not None:
        raise ValueError("Provide both --corp and --ward, or neither.")

    targets: list[tuple[int, int]] = []
    for token in WARD_TARGETS:
        try:
            corp_str, ward_str = token.strip().split("_", 1)
            targets.append((int(corp_str), int(ward_str)))
        except Exception as exc:
            raise ValueError(
                f"Invalid ward target '{token}'. Expected format like '5_40'."
            ) from exc
    if not targets:
        raise ValueError("WARD_TARGETS is empty. Add at least one entry.")
    return targets


def download_parts_for_target(
    corp: int,
    ward: int,
    start_part: int,
    out_dir: Path,
    max_consecutive_misses: int,
    retries: int,
    max_in_flight: int,
    jitter_max: float,
) -> int:
    """Download parts for one corp/ward with rolling parallel requests."""
    found = 0
    misses = 0
    next_to_submit = start_part
    next_to_consume = start_part
    stop_requested = False
    results_by_part: dict[int, tuple[Literal["ok", "miss"], str, str]] = {}
    in_flight: dict[Future[tuple[int, Literal["ok", "miss"], str, str]], int] = {}

    max_in_flight = max(1, max_in_flight)
    jitter_max = max(0.0, jitter_max)

    def submit_part(
        executor: ThreadPoolExecutor, part: int
    ) -> Future[tuple[int, Literal["ok", "miss"], str, str]]:
        filename = f"{corp}_{ward}_{part}_EPUB.pdf"
        url = BASE_URL.format(corp=corp, ward=ward, part=part)
        dest = out_dir / filename

        def runner() -> tuple[int, Literal["ok", "miss"], str, str]:
            if jitter_max > 0:
                time.sleep(random.uniform(0.0, jitter_max))
            status, reason = download_with_retries(url, dest, retries=retries)
            return part, status, reason, filename

        return executor.submit(runner)

    def fill_inflight(executor: ThreadPoolExecutor) -> None:
        nonlocal next_to_submit
        # Keep a bounded lookahead window from the current frontier.
        max_part_to_submit = next_to_consume + max_in_flight - 1
        while (
            not stop_requested
            and len(in_flight) < max_in_flight
            and next_to_submit <= max_part_to_submit
        ):
            fut = submit_part(executor, next_to_submit)
            in_flight[fut] = next_to_submit
            next_to_submit += 1

    with ThreadPoolExecutor(max_workers=max_in_flight) as executor:
        fill_inflight(executor)

        while in_flight and not stop_requested:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)

            for fut in done:
                in_flight.pop(fut, None)
                part, status, reason, filename = fut.result()
                results_by_part[part] = (status, reason, filename)

            while next_to_consume in results_by_part:
                status, reason, filename = results_by_part.pop(next_to_consume)
                if status == "ok":
                    found += 1
                    misses = 0
                else:
                    misses += 1
                    print(
                        f"miss {misses}/{max_consecutive_misses}: "
                        f"{filename} ({reason})"
                    )
                    if misses >= max_consecutive_misses:
                        stop_requested = True
                        break
                next_to_consume += 1

            if stop_requested:
                for fut in list(in_flight.keys()):
                    fut.cancel()
                break

            fill_inflight(executor)

    print(
        f"stop: reached {max_consecutive_misses} consecutive misses "
        f"for corp={corp}, ward={ward}"
    )
    print(f"done: found {found} file(s) for corp={corp}, ward={ward}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download sequential Karnataka electoral roll PDFs."
    )
    parser.add_argument("--corp", type=int, default=None, help="Corporation number")
    parser.add_argument("--ward", type=int, default=None, help="Ward number")
    parser.add_argument(
        "--start-part", type=int, default=1, help="Starting part number"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data_list_pdfs"), help="Output folder"
    )
    parser.add_argument(
        "--max-consecutive-misses",
        type=int,
        default=2,
        help="Stop after this many consecutive missing/failed parts",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per file before counting as a miss",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=10,
        help="Maximum concurrent part downloads per ward target",
    )
    parser.add_argument(
        "--jitter-max",
        type=float,
        default=0.1,
        help="Random pre-request delay in seconds (uniform 0..jitter-max)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for corp, ward in iter_targets(args.corp, args.ward):
        target_out_dir = args.out_dir / f"{corp}_{ward}"
        target_out_dir.mkdir(parents=True, exist_ok=True)
        total += download_parts_for_target(
            corp=corp,
            ward=ward,
            start_part=args.start_part,
            out_dir=target_out_dir,
            max_consecutive_misses=args.max_consecutive_misses,
            retries=args.retries,
            max_in_flight=args.max_in_flight,
            jitter_max=args.jitter_max,
        )

    print(f"all targets done: {total} file(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
