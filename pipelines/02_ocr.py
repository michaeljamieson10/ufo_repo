"""Stage 02 — OCR for scanned PDF pages.

Why this stage exists
---------------------
Stage 01 left ``needs_ocr: true`` markers on pages with no text layer.
This stage reads those markers, rasterizes the offending pages with
PyMuPDF (faster + no poppler dependency vs ``pdf2image``), runs them
through Tesseract via ``pytesseract``, and writes the OCR'd text back
into the same JSONL.

Trade-offs at a glance
----------------------
- ``tesseract`` (default) — free, CPU, ~1–3s/page on M-series Mac.
  Quality is "good enough" for declassified typewritten pages with
  decent contrast; weaker on handwriting and low-res scans.
- ``unstructured.io`` / ``Marker`` — better quality, far slower,
  heavier installs. Worth swapping in once the pipeline is wired.
- ``Claude vision`` (per-page) — best quality, but every page is an
  API call; expensive on a 4000-page corpus. Right answer for
  edge cases the OCR mangles.

Run
---
    brew install tesseract           # one-time
    python 02_ocr.py [--workers 4]

Skips pages already containing OCR'd text (idempotent).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz
from PIL import Image
import pytesseract

REPO = Path(__file__).resolve().parent.parent
PDF_DIR = REPO / "pdfs"
DATA = Path(__file__).resolve().parent / "data"
IN_PATH = DATA / "extracted.jsonl"
OUT_PATH = DATA / "extracted_ocr.jsonl"

# 200 DPI is the sweet spot for typewriter-era documents — enough
# resolution for Tesseract to be confident, not so much that single-page
# rasterization eats all your RAM.
DPI = 200


def ocr_page(pdf_path: Path, page_num: int) -> str:
    with fitz.open(pdf_path) as doc:
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        # ``--psm 6`` ("uniform block of text") tends to beat the default
        # for declassified docs because they're usually one block per
        # page rather than scattered headlines.
        return pytesseract.image_to_string(img, config="--psm 6").strip()


def _ocr_record(rec: dict) -> dict:
    pdf_path = PDF_DIR / rec["file"]
    if not pdf_path.exists():
        # Bash and Python downloads sanitized names differently — try
        # the un-sanitized variants used by the original URL.
        candidates = list(PDF_DIR.glob(rec["file"].replace("_", "*")))
        if candidates:
            pdf_path = candidates[0]
        else:
            return rec  # can't OCR, leave needs_ocr flag alone
    for p in rec["pages"]:
        if p["needs_ocr"] and not p.get("ocr_text"):
            try:
                p["ocr_text"] = ocr_page(pdf_path, p["page"])
                p["text"] = p["ocr_text"]  # promote OCR text to canonical
                p["needs_ocr"] = False
            except Exception as e:
                p["ocr_error"] = str(e)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not IN_PATH.exists():
        sys.exit(f"missing {IN_PATH}; run 01_ingest.py first")
    records = [json.loads(line) for line in IN_PATH.read_text().splitlines() if line]
    needing = [r for r in records if r.get("pages_needing_ocr", 0) > 0]
    total_pages = sum(r["pages_needing_ocr"] for r in needing)
    print(
        f"OCR queue: {len(needing)} PDFs / {total_pages} pages "
        f"with {args.workers} workers",
        file=sys.stderr,
    )

    t0 = time.time()
    by_file = {r["file"]: r for r in records}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_ocr_record, r): r["file"] for r in needing}
        for i, fut in enumerate(as_completed(futs), 1):
            updated = fut.result()
            by_file[updated["file"]] = updated
            done_pages = sum(
                1 for p in updated["pages"]
                if p.get("ocr_text") is not None or not p["needs_ocr"]
            )
            print(
                f"  {i}/{len(needing)}  {updated['file']}  ({done_pages} pages OK)",
                file=sys.stderr,
            )

    OUT_PATH.write_text(
        "\n".join(json.dumps(by_file[k]) for k in sorted(by_file)) + "\n"
    )
    print(f"Done in {time.time() - t0:.1f}s. Output: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
