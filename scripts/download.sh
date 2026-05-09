#!/usr/bin/env bash
# Mirror war.gov UAP/UFO release files locally.
# Source manifest: https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-csv.csv

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
META="$REPO_DIR/metadata"
LOG="$REPO_DIR/download.log"
: > "$LOG"

UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'

fetch() {
  local url="$1" dest="$2" referer="${3:-https://www.war.gov/ufo/}"
  if [[ -s "$dest" ]]; then
    echo "SKIP $dest" >> "$LOG"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  local code
  code=$(curl -sSL -A "$UA" \
    -H "Accept: */*" \
    -H "Accept-Language: en-US,en;q=0.9" \
    -H "Referer: $referer" \
    -H 'Sec-Ch-Ua: "Chromium";v="123", "Not:A-Brand";v="8"' \
    -H 'Sec-Ch-Ua-Mobile: ?0' \
    -H 'Sec-Ch-Ua-Platform: "macOS"' \
    -H 'Sec-Fetch-Dest: document' \
    -H 'Sec-Fetch-Mode: navigate' \
    -H 'Sec-Fetch-Site: same-origin' \
    --compressed --retry 3 --retry-delay 2 \
    -o "$dest" -w "%{http_code}" "$url")
  echo "$code $url -> $dest" >> "$LOG"
  if [[ "$code" != "200" ]]; then
    rm -f "$dest"
    return 1
  fi
}
export -f fetch
export UA LOG

echo "[1/3] PDFs ($(wc -l < "$META/pdf_urls.txt"))"
while IFS= read -r url; do
  [[ -z "$url" ]] && continue
  fname=$(basename "$url" | tr ' ' '_')
  fetch "$url" "$REPO_DIR/pdfs/$fname" &
  # throttle to ~6 concurrent
  while (( $(jobs -rp | wc -l) >= 6 )); do sleep 0.1; done
done < "$META/pdf_urls.txt"
wait

echo "[2/3] Images ($(wc -l < "$META/image_urls.txt"))"
while IFS= read -r url; do
  [[ -z "$url" ]] && continue
  fname=$(basename "$url" | tr ' ' '_')
  fetch "$url" "$REPO_DIR/images/$fname" &
  while (( $(jobs -rp | wc -l) >= 6 )); do sleep 0.1; done
done < "$META/image_urls.txt"
wait

echo "[3/3] Videos ($(wc -l < "$META/dvids_video_ids.txt"))"
while IFS= read -r vid; do
  [[ -z "$vid" ]] && continue
  page="$REPO_DIR/videos/.page_${vid}.html"
  curl -sSL -A "$UA" -H "Accept: text/html" --compressed \
    "https://www.dvidshub.net/video/$vid" -o "$page"
  mp4=$(grep -oE 'https?://[^"'\''" ]+\.mp4[^"'\''" ]*' "$page" | head -1)
  if [[ -n "$mp4" ]]; then
    fname="dvids_${vid}_$(basename "${mp4%%\?*}")"
    fetch "$mp4" "$REPO_DIR/videos/$fname" "https://www.dvidshub.net/video/$vid"
  else
    echo "NO_MP4 $vid" >> "$LOG"
  fi
  rm -f "$page"
done < "$META/dvids_video_ids.txt"

echo "Done. See $LOG"
