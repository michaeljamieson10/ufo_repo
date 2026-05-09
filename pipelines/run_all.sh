#!/usr/bin/env bash
# End-to-end pipeline driver.
#
# Order matters:
#   01 → 02 → 03 must run sequentially (each consumes the previous output).
#   04 is library code (no side effects — used by 05).
#   05 is the user-facing entrypoint (a question goes here).
#   06 is optional/stretch.

set -euo pipefail
cd "$(dirname "$0")"

echo "[01] PDF ingestion + scan classifier"
python 01_ingest.py

echo "[02] OCR scanned pages (skip if you don't have tesseract installed)"
if command -v tesseract >/dev/null 2>&1; then
  python 02_ocr.py --workers 4
else
  echo "  tesseract not found — skipping. brew install tesseract to enable."
  cp data/extracted.jsonl data/extracted_ocr.jsonl
fi

echo "[03] Chunk + embed → Chroma"
python 03_chunk_embed.py

echo
echo "Ready. Try:"
echo "  python 04_retrieve.py 'what does the FBI know about Roswell?'"
echo "  python 05_agent.py    'compare 1947 and 2024 sightings'"
