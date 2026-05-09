"""Cross-release file discovery + path resolution.

The corpus is rolling — Department of War said new tranches will be
released over time. We mirror each one into ``releases/release_NN/``
with the same internal layout (pdfs/ images/ videos/ metadata/).

Pipeline scripts should never hard-code ``ufo_repo/pdfs`` or similar;
import from here instead so a new release becomes a no-op for the
ingestion code.

Usage
-----
    from scripts.corpus import all_pdfs, all_videos, all_images, releases

    for pdf_path, release_id in all_pdfs():
        ...

Each helper yields ``(path, release_id)`` so per-document metadata can
record which release the file came from. That lets retrieval answer
"what's new in release_02?" out of the box.
"""
from __future__ import annotations

import csv
import os
import re
import urllib.parse
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent.parent
RELEASES_DIR = REPO / "releases"


def releases() -> list[Path]:
    """All release directories, sorted by release number."""
    if not RELEASES_DIR.exists():
        return []
    return sorted(
        (p for p in RELEASES_DIR.iterdir() if p.is_dir() and p.name.startswith("release_")),
        key=lambda p: int(re.search(r"\d+", p.name).group()),
    )


def release_id(release_dir: Path) -> str:
    """Stable id for a release dir — used as a metadata tag on chunks."""
    return release_dir.name  # e.g. "release_01"


def _iter_glob(pattern: str) -> Iterator[tuple[Path, str]]:
    for r in releases():
        for p in sorted((r / pattern.split("/")[0]).glob(pattern.split("/", 1)[1] if "/" in pattern else "*")):
            if p.is_file():
                yield p, release_id(r)


def all_pdfs() -> Iterator[tuple[Path, str]]:
    for r in releases():
        d = r / "pdfs"
        if not d.exists():
            continue
        for p in sorted(d.glob("*.pdf")):
            yield p, release_id(r)


def all_images() -> Iterator[tuple[Path, str]]:
    for r in releases():
        d = r / "images"
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}:
                yield p, release_id(r)


def all_videos() -> Iterator[tuple[Path, str]]:
    for r in releases():
        d = r / "videos"
        if not d.exists():
            continue
        for p in sorted(d.glob("*.mp4")):
            yield p, release_id(r)


def all_manifests() -> Iterator[tuple[Path, str]]:
    """The war.gov uap-csv.csv per release."""
    for r in releases():
        m = r / "metadata" / "uap-csv.csv"
        if m.exists():
            yield m, release_id(r)


def manifest_rows() -> Iterator[tuple[dict, str]]:
    """Iterate every row across every release's CSV manifest, tagged with release id."""
    for path, rid in all_manifests():
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                yield row, rid


def basename_from_url(url: str) -> str:
    return urllib.parse.unquote(os.path.basename(urllib.parse.urlsplit(url).path))


def safe_name(name: str) -> str:
    name = name.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._\-\[\]]", "_", name)


def find_pdf(release_id_: str, csv_basename: str) -> Path | None:
    """Resolve a CSV-listed PDF basename against the on-disk file in
    its release dir, tolerating sanitization differences from the
    downloader (em-dashes, brackets, apostrophes)."""
    d = RELEASES_DIR / release_id_ / "pdfs"
    for cand in (csv_basename, safe_name(csv_basename), csv_basename.replace(" ", "_")):
        p = d / cand
        if p.exists():
            return p
    return None


__all__ = [
    "REPO",
    "RELEASES_DIR",
    "all_images",
    "all_manifests",
    "all_pdfs",
    "all_videos",
    "basename_from_url",
    "find_pdf",
    "manifest_rows",
    "release_id",
    "releases",
    "safe_name",
]
