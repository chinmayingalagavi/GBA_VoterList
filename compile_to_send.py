"""Compile per-ward CSV bundles from output_list_pdfs into to_send.

For each top-level ward folder under input_root:
- Merge all CSVs in that ward folder.
- Derive ward label from cover_metadata.json files under ward subfolders.
- Write one CSV to output_dir named:
    "<ward_key> <ward_label>.csv"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

# Corporation number to canonical name used in output filenames.

# central = 1
# north = 2
# south = 3
# east = 4
# west = 5
CORPORATIONS = {1: "Central", 2: "North", 3: "South", 4: "East", 5: "West"}
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


def natural_key(text: str) -> list[object]:
    """Natural sort helper: '..._2' comes before '..._10'."""
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe filename stem."""
    safe = re.sub(r'[<>:"/\\|?*]+', "_", name)
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe or "combined"


def excel_text_apostrophe_space(value: str) -> str:
    """Force Excel to treat a field as text via apostrophe + space prefix."""
    text = str(value or "")
    if not text:
        return text
    if text.startswith("' "):
        return text
    return f"' {text}"


def normalize_gender(raw: str) -> str:
    """Normalize gender values to 'M' or 'F'."""
    val = str(raw or "").strip().upper()
    if val in ("M", "MALE"):
        return "M"
    if val in ("F", "FEMALE"):
        return "F"
    return str(raw or "")


def detect_ward_label(ward_dir: Path) -> str:
    """Pick the most common non-empty `ward` value from cover metadata files."""
    values: list[str] = []
    for cover_path in sorted(ward_dir.glob("*/cover_metadata.json")):
        try:
            data = json.loads(cover_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ward = str(data.get("ward", "")).strip()
        if ward:
            values.append(ward)

    if not values:
        return ward_dir.name

    counts = Counter(values)
    ward, _ = counts.most_common(1)[0]
    if len(counts) > 1:
        print(
            f"warning: multiple ward labels in {ward_dir.name}, "
            f"using most common: {ward!r}"
        )
    return ward


def corporation_name_from_ward_key(ward_key: str) -> str:
    """Resolve corp direction label from ward key like '5_40'."""
    try:
        corp_no = int(ward_key.split("_", 1)[0])
    except Exception:
        return "UnknownCorp"
    return CORPORATIONS.get(corp_no, "UnknownCorp")


def merge_ward_csvs(ward_dir: Path, output_dir: Path) -> tuple[Path, int, int]:
    """Build one ward CSV from per-PDF cached page JSON + cover metadata."""
    pdf_dirs = sorted(
        [p for p in ward_dir.iterdir() if p.is_dir() and p.name.endswith("_EPUB")],
        key=lambda p: natural_key(p.name),
    )
    if not pdf_dirs:
        raise ValueError(f"No PDF cache folders found in {ward_dir}")

    ward_label = detect_ward_label(ward_dir).strip()
    corporation_name = corporation_name_from_ward_key(ward_dir.name)
    out_stem = sanitize_filename(
        f"{corporation_name} {ward_label} {ward_dir.name}"
    )
    out_path = output_dir / f"{out_stem}.csv"
    tmp_path = out_path.with_suffix(".csv.tmp")

    total_rows = 0
    source_count = 0

    with open(tmp_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for pdf_dir in pdf_dirs:
            cover_path = pdf_dir / "cover_metadata.json"
            pages_dir = pdf_dir / "pages"

            if not cover_path.exists() or not pages_dir.exists():
                continue

            try:
                cover = json.loads(cover_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"warning: skipping {pdf_dir.name} (invalid cover metadata: {exc})")
                continue

            page_paths = sorted(
                pages_dir.glob("page_*.json"), key=lambda p: natural_key(p.name)
            )
            if not page_paths:
                continue

            source_count += 1
            for page_path in page_paths:
                try:
                    page_data = json.loads(page_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"warning: skipping invalid page JSON {page_path}: {exc}")
                    continue

                section_raw = str(page_data.get("section_raw", ""))
                voters = page_data.get("voters", [])
                if not isinstance(voters, list):
                    continue

                for voter in voters:
                    if not isinstance(voter, dict):
                        continue
                    row = {
                        "serial_number": voter.get("serial_number", ""),
                        "epic_number": voter.get("voter_id_epic", ""),
                        "name": voter.get("name_english", ""),
                        "relation_type": voter.get("relationship_type", ""),
                        "relation_name": voter.get("relative_name_english", ""),
                        "house_number": excel_text_apostrophe_space(
                            voter.get("house_number", "")
                        ),
                        "age": voter.get("age", ""),
                        "gender": normalize_gender(str(voter.get("gender", ""))),
                        "section_raw": section_raw,
                        "part": cover.get("part", ""),
                        "ward": cover.get("ward", ""),
                        "corporation": cover.get("corporation", ""),
                        "pincode": cover.get("pincode", ""),
                        "pollingname": cover.get("pollingname", ""),
                        "pollingaddress": cover.get("pollingaddress", ""),
                    }
                    writer.writerow(row)
                    total_rows += 1

    tmp_path.replace(out_path)
    return out_path, source_count, total_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile one combined CSV per top-level ward folder from output_list_pdfs."
        )
    )
    parser.add_argument(
        "--input-root", type=Path, default=Path("output_list_pdfs")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("to_send")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ward_dirs = sorted(
        [p for p in args.input_root.iterdir() if p.is_dir()],
        key=lambda p: natural_key(p.name),
    )
    if not ward_dirs:
        print(f"No ward folders found in {args.input_root}")
        return

    for ward_dir in ward_dirs:
        try:
            out_path, source_count, row_count = merge_ward_csvs(
                ward_dir, args.output_dir
            )
            print(
                f"{ward_dir.name}: merged {source_count} files "
                f"into {out_path} ({row_count} rows)"
            )
        except ValueError as exc:
            print(f"{ward_dir.name}: skipped ({exc})")


if __name__ == "__main__":
    main()
