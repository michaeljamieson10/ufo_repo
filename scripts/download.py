#!/usr/bin/env python3
"""Mirror a war.gov UAP/UFO release into ``releases/release_NN/``.

The Department of War said this is a rolling release. Each tranche
publishes its own manifest CSV; new files are appended to a release
directory keyed by the URL's ``release_N`` segment so the corpus stays
forward-compatible.

Defaults to release 1 with the known manifest URL. For future releases
pass ``--release N`` and either ``--manifest-url ...`` or rely on the
default URL pattern (war.gov keeps the same path style across releases).

Akamai blocks bare requests; we mimic Chrome client hints to get 200s.
DVIDS video pages embed a CloudFront mp4 URL we can scrape.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RELEASES_DIR = REPO / "releases"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
BASE_HEADERS = {
    "User-Agent": UA,
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="123", "Not:A-Brand";v="8"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Upgrade-Insecure-Requests": "1",
}

# Known manifest URLs by release. Add a row when a new tranche drops.
KNOWN_MANIFESTS = {
    1: "https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-csv.csv",
}


def encode_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(parts.path, safe="/%-_.~")
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, path, parts.query, parts.fragment)
    )


def fetch(url: str, dest: Path, log_fp, referer: str = "https://www.war.gov/ufo/") -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return f"SKIP {dest.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    enc = encode_url(url)
    headers = dict(BASE_HEADERS)
    headers["Referer"] = referer
    req = urllib.request.Request(enc, headers=headers)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
        size = dest.stat().st_size
        return f"OK {size:>12d}  {dest.name}"
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return f"FAIL {url} :: {e}"


def safe_name(url: str) -> str:
    """Preserve the original URL filename verbatim (after %-decoding and
    space → underscore). The CSV manifest sometimes lists the same file
    under two URL spellings (em-dash vs hyphen, with/without brackets,
    apostrophe vs underscore); preserving the original characters means
    re-runs hit the existing on-disk file instead of writing a sanitized
    duplicate."""
    name = os.path.basename(urllib.parse.urlsplit(url).path)
    name = urllib.parse.unquote(name).replace(" ", "_")
    # Only strip characters that are illegal on the local filesystem.
    return re.sub(r"[\x00-\x1f<>:\"|?*]", "_", name)


def fetch_manifest(release_dir: Path, manifest_url: str) -> Path:
    out = release_dir / "metadata" / "uap-csv.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return out
    headers = dict(BASE_HEADERS)
    headers["Referer"] = "https://www.war.gov/ufo/"
    req = urllib.request.Request(encode_url(manifest_url), headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        out.write_bytes(resp.read())
    print(f"  manifest -> {out}", file=sys.stderr)
    return out


def parse_manifest(csv_path: Path) -> tuple[list[str], list[str], list[str], list[dict]]:
    """Return (pdfs, images, dvids_ids, raw_rows). Dedups within each."""
    pdfs: set[str] = set()
    images: set[str] = set()
    dvids: set[str] = set()
    rows: list[dict] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            pl = (r.get("PDF | Image Link") or "").strip()
            mi = (r.get("Modal Image") or "").strip()
            vi = (r.get("DVIDS Video ID") or "").strip()
            if pl.startswith("http"):
                if pl.lower().endswith(".pdf"):
                    pdfs.add(pl)
                elif pl.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                    images.add(pl)
            if mi.startswith("http"):
                images.add(mi)
            if vi.isdigit():
                dvids.add(vi)
    return sorted(pdfs), sorted(images), sorted(dvids), rows


def get_dvids_mp4(video_id: str, log_fp) -> str | None:
    page_url = f"https://www.dvidshub.net/video/{video_id}"
    headers = dict(BASE_HEADERS)
    headers["Referer"] = "https://www.war.gov/ufo/"
    headers["Sec-Fetch-Site"] = "cross-site"
    req = urllib.request.Request(page_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        log_fp.write(f"DVIDS_PAGE_FAIL {video_id} :: {e}\n")
        return None
    m = re.search(r'https?://[^"\' ]+\.mp4[^"\' ]*', html)
    return m.group(0) if m else None


def download_release(release_num: int, manifest_url: str | None) -> None:
    release_dir = RELEASES_DIR / f"release_{release_num:02d}"
    release_dir.mkdir(parents=True, exist_ok=True)

    log_path = release_dir / "download.log"
    log_fp = open(log_path, "a", buffering=1)
    log_fp.write(f"=== run release_{release_num:02d} {time.ctime()} ===\n")

    url = manifest_url or KNOWN_MANIFESTS.get(release_num)
    if not url:
        sys.exit(
            f"no manifest URL for release {release_num}. "
            f"Pass --manifest-url, or add it to KNOWN_MANIFESTS."
        )
    print(f"[release_{release_num:02d}] manifest: {url}", file=sys.stderr)
    csv_path = fetch_manifest(release_dir, url)

    pdfs, images, dvids, rows = parse_manifest(csv_path)
    (release_dir / "metadata" / "manifest.json").write_text(json.dumps(rows, indent=2))
    (release_dir / "metadata" / "pdf_urls.txt").write_text("\n".join(pdfs) + "\n")
    (release_dir / "metadata" / "image_urls.txt").write_text("\n".join(images) + "\n")
    (release_dir / "metadata" / "dvids_video_ids.txt").write_text("\n".join(dvids) + "\n")

    print(f"  PDFs={len(pdfs)} images={len(images)} videos={len(dvids)}", file=sys.stderr)

    pdf_dir = release_dir / "pdfs"
    img_dir = release_dir / "images"
    vid_dir = release_dir / "videos"

    print(f"[1/3] PDFs", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch, u, pdf_dir / safe_name(u), log_fp): u for u in pdfs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            log_fp.write(res + "\n")
            if done % 10 == 0 or done == len(pdfs):
                print(f"  pdfs {done}/{len(pdfs)}", file=sys.stderr)

    print(f"[2/3] Images", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch, u, img_dir / safe_name(u), log_fp): u for u in images}
        done = 0
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            log_fp.write(res + "\n")
            if done % 20 == 0 or done == len(images):
                print(f"  images {done}/{len(images)}", file=sys.stderr)

    print(f"[3/3] Videos (DVIDS)", file=sys.stderr)
    for i, vid in enumerate(sorted(dvids), 1):
        mp4 = get_dvids_mp4(vid, log_fp)
        if not mp4:
            print(f"  videos {i}/{len(dvids)} NO_MP4 {vid}", file=sys.stderr)
            continue
        fname = f"dvids_{vid}_{safe_name(mp4.split('?')[0])}"
        res = fetch(mp4, vid_dir / fname, log_fp,
                    referer=f"https://www.dvidshub.net/video/{vid}")
        log_fp.write(res + "\n")
        print(f"  videos {i}/{len(dvids)}  {res[:80]}", file=sys.stderr)

    dedup_release(release_dir, log_fp)

    log_fp.close()
    print(f"\nDone. See {log_path}", file=sys.stderr)


def dedup_release(release_dir: Path, log_fp) -> None:
    """Remove byte-identical files in pdfs/ and images/.

    The CSV manifest sometimes lists the same asset under two URL
    spellings (em-dash vs hyphen, apostrophe vs underscore, etc.) and
    the server returns the same bytes for both. We keep the spelling
    closest to the original URL (most non-alphanumeric chars beyond _).
    """
    import hashlib
    from collections import defaultdict

    for sub in ("pdfs", "images"):
        d = release_dir / sub
        if not d.exists():
            continue
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for p in d.iterdir():
            if not p.is_file():
                continue
            h = hashlib.md5(p.read_bytes()).hexdigest()
            by_hash[h].append(p)
        for paths in by_hash.values():
            if len(paths) < 2:
                continue
            keep = max(paths, key=lambda p: sum(c in "[]'’–,;()" for c in p.name))
            for p in paths:
                if p != keep:
                    p.unlink()
                    log_fp.write(f"DEDUP_REMOVED {p}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=int, default=1,
                    help="Release number (1, 2, 3, ...).")
    ap.add_argument("--manifest-url", default=None,
                    help="Override manifest URL. Required for unknown releases.")
    args = ap.parse_args()
    download_release(args.release, args.manifest_url)


if __name__ == "__main__":
    main()
