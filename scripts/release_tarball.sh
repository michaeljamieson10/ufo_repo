#!/usr/bin/env bash
# Package the built DBs into a tarball for GitHub Releases.
#
# Why publish DBs separately from the repo: SQLite (Chroma) and the
# FalkorDB Redis dump rewrite on every change, so committing them
# would balloon git history. A tarball asset on GitHub Releases is
# what most production "shareable RAG" projects use.
#
# Usage:
#   scripts/release_tarball.sh                         # auto-versioned
#   scripts/release_tarball.sh ufo-dbs-2026-05-09.tar.gz
#
# After running, upload the tarball to a GitHub Release:
#   gh release create v$(date +%Y.%m.%d) ufo-dbs-*.tar.gz \
#     --title "Pre-built DBs $(date +%Y-%m-%d)" \
#     --notes "Chroma + FalkorDB built from release_01 + ..."

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

NAME="${1:-ufo-dbs-$(date +%Y-%m-%d).tar.gz}"

# Force a fresh FalkorDB save so the on-host RDB is current. If the
# host volume mount doesn't have the RDB (older docker-compose mount
# pointed at /data instead of /var/lib/falkordb/data), copy it out
# of the container directly.
if docker ps --format '{{.Names}}' | grep -q '^ufo-falkordb$'; then
  echo "Flushing FalkorDB to disk..."
  docker exec ufo-falkordb redis-cli BGSAVE >/dev/null
  sleep 2
  mkdir -p pipelines/graphrag/data/falkordb
  if [ ! -f pipelines/graphrag/data/falkordb/dump.rdb ]; then
    echo "Copying dump.rdb out of container (mount-path mismatch fallback)..."
    docker cp ufo-falkordb:/var/lib/falkordb/data/dump.rdb \
      pipelines/graphrag/data/falkordb/dump.rdb
  fi
fi

# What goes in: the binary stores. What stays out: source PDFs/videos
# (those are mirrorable from war.gov), virtualenvs, frame jpgs (huge),
# .DS_Store noise.
tar --exclude='*/__pycache__' \
    --exclude='*/.DS_Store' \
    -czf "$NAME" \
    pipelines/store/ \
    pipelines/graphrag/data/falkordb/ \
    pipelines/data/extracted_ocr.jsonl \
    pipelines/data/extracted.jsonl \
    pipelines/graphrag/data/*.jsonl 2>/dev/null || true

ls -lh "$NAME"
echo "Done. Upload to GitHub Releases via:"
echo "  gh release create v\$(date +%Y.%m.%d) $NAME --notes-file release_notes.md"
