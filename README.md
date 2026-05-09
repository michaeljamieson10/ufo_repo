# UFO Repo — war.gov UAP Release Mirror

Local mirror of the U.S. Department of War's UAP/UFO file release.

Source: https://www.war.gov/ufo/
Manifest: https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-csv.csv
Mirrored: 2026-05-09

## Contents

| Directory | Files | Size  | Notes |
|-----------|------:|------:|-------|
| `pdfs/`   |   115 |  2.5G | FBI/DoD/agency case files in PDF |
| `images/` |   138 |   42M | PDF thumbnails + slideshow imagery |
| `videos/` |    28 |  1.3G | DVIDS-hosted UAP encounter videos (mp4) |
| `metadata/` | 5  |  448K | Original CSV, JSON manifest, URL lists |

Total: ~3.8 GB.

## metadata/

- `uap-csv.csv` — original manifest from war.gov (161 entries)
- `manifest.json` — same data parsed to JSON for easy use
- `pdf_urls.txt`, `image_urls.txt`, `dvids_video_ids.txt` — extracted URL lists

The CSV columns include: Title, Type (PDF/VID/IMG), Description, Agency, Incident Date, Incident Location, DVIDS Video ID, PDF/Image link, etc.

## scripts/download.py

Re-runnable Python downloader. Skips files that already exist, so it's safe to rerun if war.gov adds more entries.

```bash
python3 scripts/download.py
```

Notes on bypassing Akamai: war.gov returns 403 to plain `curl`/`wget`; the downloader sends Chrome client-hint headers (`Sec-Ch-Ua*`, `Sec-Fetch-*`) which lets requests through. DVIDS video pages are scraped for the embedded CloudFront mp4 URL.

## download.log

Append-only log of every fetch (`OK <bytes> <name>` or `FAIL <url> :: <error>`). Last run: 0 failures.
