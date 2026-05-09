#!/usr/bin/env bash
# Rebuild Chroma + FalkorDB from the committed JSONL artifacts.
#
# This is the "cheap path" for downstream users — no LLM time required,
# everything's already extracted. Just runs the local-only steps:
# chunking + embedding (Chroma) and graph loading (FalkorDB).
#
# Expected runtime on an M-series Mac: ~5 minutes total.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [ ! -d .venv ]; then
  echo "Setting up venv..."
  python3.11 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r pipelines/requirements.txt
  .venv/bin/pip install -q -r pipelines/graphrag/requirements.txt
fi

PY="$REPO/.venv/bin/python"

echo "[1/2] Chroma — chunk + embed from pipelines/data/extracted_ocr.jsonl"
if [ ! -f pipelines/data/extracted_ocr.jsonl ] && [ ! -f pipelines/data/extracted.jsonl ]; then
  echo "  No extracted JSONL found. Running ingest first..."
  "$PY" pipelines/01_ingest.py
  if command -v tesseract >/dev/null 2>&1; then
    "$PY" pipelines/02_ocr.py --workers 6
  else
    echo "  WARNING: tesseract not installed — only text-layer PDFs will be indexed."
    echo "           brew install tesseract for full OCR."
  fi
fi
"$PY" pipelines/03_chunk_embed.py

echo
echo "[2/2] FalkorDB — load entities + communities + frames"
if ! docker ps --format '{{.Names}}' | grep -q '^ufo-falkordb$'; then
  echo "  Starting FalkorDB container..."
  (cd pipelines/graphrag && docker compose up -d)
  sleep 4
fi
if [ -f pipelines/graphrag/data/chunk_entities.jsonl ]; then
  "$PY" pipelines/graphrag/05_load_falkordb.py
else
  echo "  No chunk_entities.jsonl committed — skipping FalkorDB load."
  echo "  Run pipelines/graphrag/run_all.sh to build the GraphRAG layer first."
fi

echo
echo "Done. Try:"
echo "  $PY pipelines/05_agent.py 'what does the FBI know about flying discs?'"
echo "  $PY pipelines/graphrag/06_graph_agent.py 'incidents in 1947 with named witnesses'"
