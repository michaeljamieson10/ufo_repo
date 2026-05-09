# Release 01 — 2026-05-08

First tranche of the U.S. Department of War's PURSUE UAP/UFO release.
Mirrored on 2026-05-09.

- Press release: https://www.war.gov/News/Releases/Release/Article/4480582/
- Source page: https://www.war.gov/UFO/
- Manifest: https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-csv.csv

## Contents

| Directory | Files | Size  | Notes |
|-----------|------:|------:|-------|
| `pdfs/`   |   115 |  2.4G | FBI / DoD / NASA / NARA / DOS case files |
| `images/` |   138 |   42M | 14 primary IMG-type + 124 PDF thumbnails |
| `videos/` |    28 |  1.3G | DVIDS-hosted UAP encounter videos (mp4) |

## Composition (per the manifest)

| Type | Count |
|------|------:|
| PDF | 119 manifest rows → 115 unique files |
| VID | 28 |
| IMG | 14 |

| Agency | Approx file count |
|--------|------:|
| FBI | ~57 |
| DOW (Dept of War) | ~44 |
| NASA | ~13 |
| NARA | ~13 |
| DOS | ~5 |

## Notes

- The manifest occasionally lists the same asset under two URL
  spellings (em-dash vs hyphen, apostrophe vs underscore). The
  downloader's content-hash dedup keeps the spelling closest to the
  original URL.
- DVIDS videos vary in length from a few seconds to ~5 minutes;
  total duration ~1 hr.
- About half the PDFs are scanned (no text layer); pipeline stage 02
  OCRs them.
