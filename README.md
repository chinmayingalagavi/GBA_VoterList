# Electoral Roll PDF Parser (Karnataka SEC)

Parse Karnataka electoral roll PDFs into structured ward-level CSVs using Gemini vision extraction.

## What This Repo Does

- Downloads electoral roll PDFs by corp/ward/part pattern.
- Parses each PDF page into canonical JSON cache files.
- Extracts cover-page metadata (part, ward, corporation, polling details, counts).
- Compiles final ward-level CSVs for sharing.

## Current Pipeline

1. `list_downloader.py`  
   Downloads PDFs into `data_list_pdfs/<corp>_<ward>/`.
   The first list in list_downloader needs to be edited first, to choose which wards to target.

2. `parse_doc.py`  
   Reads PDFs recursively from `data_list_pdfs/`, extracts data, writes:
   - `output_list_pdfs/<corp>_<ward>/<pdf_stem>/cover_metadata.json`
   - `output_list_pdfs/<corp>_<ward>/<pdf_stem>/pages/page_XXX.json`
   - optional per-PDF CSV: `output_list_pdfs/<corp>_<ward>/<pdf_stem>.csv`

3. `compile_to_send.py`  
   Builds final ward-level CSVs in `to_send/` directly from cached JSON files.

## Requirements

- Python 3.10+
- Gemini API key(s)
- Packages:
  - `google-genai`
  - `python-dotenv`
  - `PyPDF2`
  - `pydantic`

Install:

```bash
pip install google-genai python-dotenv PyPDF2 pydantic
```

## Environment Setup

Create `.env` in repo root:

```env
# Use one key:
GEMINI_API_KEY=your_key_here

# Or rotate across many keys (comma or newline separated):
# GEMINI_API_KEYS=key1,key2,key3

# Optional parser tuning:
# MAX_CONCURRENT=200
# PDF_PREFETCH=15
# INPUT_DIR=data_list_pdfs
# OUTPUT_DIR=output_list_pdfs
# LOG_ROOT_DIR=logs
```

Very slow if you are not doing many at a time!

Notes:
- If `GEMINI_API_KEYS` is set, parser rotates keys.
- `MAX_CONCURRENT` is global target concurrency; per-active-PDF page concurrency is derived from it.

## Usage

### 1) Download PDFs

Edit `WARD_TARGETS` in `list_downloader.py`, then run:

```bash
python list_downloader.py
```

Useful flags:

```bash
python list_downloader.py \
  --max-in-flight 10 \
  --max-consecutive-misses 2 \
  --retries 3 \
  --jitter-max 0.1 \
  --start-part 1 \
  --out-dir data_list_pdfs
```

### 2) Parse PDFs into canonical JSON cache

```bash
python parse_doc.py
```

Parser behavior:
- Skips page 2 (maps/photos); processes page 3 onward.
- Uses existing cached page JSON to resume cleanly.
- Re-calls cover extraction only if `cover_metadata.json` is missing/invalid.
- Input discovery is recursive and only matches `*.pdf` (not `.pdf.tmp`).

### 3) Build final ward CSVs

```bash
python compile_to_send.py
```

Optional paths:

```bash
python compile_to_send.py --input-root output_list_pdfs --output-dir to_send
```

Output filename format:

`<CorporationName> <WardLabel> <corpno_wardno>.csv`

Example:

`West 40 - Mahalakshmipuram 5_40.csv`

## Data Model (final CSV columns)

```text
serial_number, epic_number, name, relation_type, relation_name, house_number,
age, gender, section_raw, part, ward, corporation, pincode, pollingname, pollingaddress
```

I used `gemini-flash-3-preview` to do the lists and `gemini-pro-3-preview` to do the cover page, since I wanted very high accuracy for the main street address from the cover.

Excel note:
- `compile_to_send.py` prefixes `house_number` with `' ` (apostrophe + space) to prevent date auto-conversion (e.g., `2-3`).

## Directory Layout

```text
.
├── data_list_pdfs/        # downloaded source PDFs by ward folder
├── output_list_pdfs/      # canonical parsed outputs (cover + per-page JSON)
├── to_send/               # final merged ward CSVs
├── list_downloader.py
├── parse_doc.py
├── compile_to_send.py
├── gemini_client.py
└── .env
```

## Troubleshooting

- Slow runs are usually API-side latency/rate limits (429/5xx retries use backoff + jitter).
- If a PDF fails with exhausted pages, rerun `python parse_doc.py`; completed page caches are reused.
- If cover metadata looks wrong, delete only that PDF’s `cover_metadata.json` and rerun parser.


### Acknowledgements
Vibecoded with Claude Code Opus 4.6 and Codex 5.3.

