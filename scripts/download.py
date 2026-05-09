#!/usr/bin/env python3
"""Mirror war.gov UAP/UFO release files locally.

Source manifest: https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-csv.csv
Akamai blocks bare requests; we mimic Chrome client hints to get 200s.
DVIDS video pages embed a CloudFront mp4 URL we can scrape.
"""
from __future__ import annotations

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
META = REPO / "metadata"
LOG = REPO / "download.log"

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

log_lock_path = LOG
log_fp = open(LOG, "a", buffering=1)


def log(msg: str) -> None:
    log_fp.write(msg + "\n")


def encode_url(url: str) -> str:
    """URL-encode the path portion only (preserve scheme/host/query)."""
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(parts.path, safe="/%-_.~")
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, path, parts.query, parts.fragment)
    )


def fetch(url: str, dest: Path, referer: str = "https://www.war.gov/ufo/") -> str:
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
    name = os.path.basename(urllib.parse.urlsplit(url).path)
    name = urllib.parse.unquote(name).replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._\-\[\]]", "_", name)


def download_list(urls: list[str], outdir: Path, label: str, workers: int = 8) -> None:
    urls = sorted(set(u for u in urls if u))
    print(f"[{label}] {len(urls)} files -> {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, u, outdir / safe_name(u)): u for u in urls}
        done = 0
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            log(res)
            if done % 10 == 0 or done == len(urls):
                print(f"  [{label}] {done}/{len(urls)}  {res[:80]}")


def get_dvids_mp4(video_id: str) -> str | None:
    page_url = f"https://www.dvidshub.net/video/{video_id}"
    headers = dict(BASE_HEADERS)
    headers["Referer"] = "https://www.war.gov/ufo/"
    headers["Sec-Fetch-Site"] = "cross-site"
    req = urllib.request.Request(page_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        log(f"DVIDS_PAGE_FAIL {video_id} :: {e}")
        return None
    m = re.search(r'https?://[^"\' ]+\.mp4[^"\' ]*', html)
    return m.group(0) if m else None


def download_videos(ids: list[str], outdir: Path) -> None:
    ids = sorted(set(i for i in ids if i))
    print(f"[VIDEOS] {len(ids)} DVIDS ids -> {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    for i, vid in enumerate(ids, 1):
        mp4 = get_dvids_mp4(vid)
        if not mp4:
            log(f"NO_MP4 {vid}")
            print(f"  [VIDEOS {i}/{len(ids)}] NO_MP4 {vid}")
            continue
        fname = f"dvids_{vid}_{safe_name(mp4.split('?')[0])}"
        res = fetch(mp4, outdir / fname, referer=f"https://www.dvidshub.net/video/{vid}")
        log(res)
        print(f"  [VIDEOS {i}/{len(ids)}] {res[:90]}")


def main() -> None:
    pdfs = (META / "pdf_urls.txt").read_text().splitlines()
    images = (META / "image_urls.txt").read_text().splitlines()
    vids = (META / "dvids_video_ids.txt").read_text().splitlines()

    download_list(pdfs, REPO / "pdfs", "PDFS", workers=8)
    download_list(images, REPO / "images", "IMG", workers=8)
    download_videos(vids, REPO / "videos")
    print("Done.")


if __name__ == "__main__":
    main()
