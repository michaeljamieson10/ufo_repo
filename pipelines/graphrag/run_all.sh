#!/usr/bin/env bash
# Build the entire GraphRAG knowledge base from the war.gov UAP corpus.
#
# Order matters; each stage consumes the previous one's output.
# Times below are M-series Mac estimates with the local Claude CLI.

set -euo pipefail
cd "$(dirname "$0")"

# Pre-req: PDF text extraction (uses ../01_ingest.py + ../02_ocr.py)
if [ ! -f ../data/extracted_ocr.jsonl ] && [ ! -f ../data/extracted.jsonl ]; then
  echo "[pre] extracting PDF text first"
  (cd .. && python 01_ingest.py)
  if command -v tesseract >/dev/null 2>&1; then
    (cd .. && python 02_ocr.py --workers 4)
  fi
fi

# 1) FalkorDB
if ! docker ps --format '{{.Names}}' | grep -q '^ufo-falkordb$'; then
  echo "[falkordb] starting container"
  docker compose up -d
  sleep 3
fi

# 2) Video keyframes — ~20 min for 28 videos
echo "[01] video keyframes (Claude vision + CLIP)"
python 01_video_frames.py

# 3) Image captions — ~5 min for 14 primaries
echo "[02] image captions"
python 02_image_caption.py

# 4) Entity + incident extraction — biggest stage, ~1-2 hr
echo "[03] entity extraction"
python 03_entity_extract.py --workers 6

# 5) Community detection + summaries — ~10 min
echo "[04] community detection + summaries"
python 04_communities.py

# 6) Push everything into FalkorDB
echo "[05] load FalkorDB"
python 05_load_falkordb.py

echo
echo "Done. Try a question:"
echo "  python 06_graph_agent.py 'which agencies reported tic-tac shaped objects?'"
echo "  python 06_graph_agent.py --mode global 'what are the major themes in the FBI files?'"
echo "  python 06_graph_agent.py --mode structured 'incidents in 1947 with named witnesses'"
echo
echo "Browse the graph at http://localhost:3000"
