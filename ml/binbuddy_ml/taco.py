"""TACO download and cropping, robust to dead Flickr links (WS1, used by notebook 01).

TACO's own download.py stops at the first dead link and stops at the same link on every rerun.
On 14 Oct 2026, 20 of 30 sampled original-size links failed, while original-or-640px succeeded
for 40 of 40. So each photo is fetched from the original link, then the 640 px link, and failures
are counted instead of aborting. Boxes are annotated on the original photo, so they are scaled to
whichever version was downloaded.
"""

from __future__ import annotations

import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

USER_AGENT = "Mozilla/5.0 (BinBuddy research; NUS-ISS PRS)"
JPEG_MAGIC = b"\xff\xd8\xff"


def _fetch(url: str, timeout: float = 30.0) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data: bytes = r.read()
    except Exception:  # dead link, timeout, HTTP error: all mean "try the next URL"
        return None
    return data if data[:3] == JPEG_MAGIC or data[:8] == b"\x89PNG\r\n\x1a\n" else None


def download_one(meta: Mapping[str, Any], root: Path) -> str:
    """Download one photo to root/<file_name>: 'exists', 'original', '640' or 'failed'."""
    out = root / str(meta["file_name"])
    if out.exists():
        return "exists"
    for kind, key in (("original", "flickr_url"), ("640", "flickr_640_url")):
        url = meta.get(key)
        if url and (data := _fetch(str(url))):
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)  # raw bytes: keeps EXIF orientation for oriented() below
            return kind
    return "failed"


def download_all(
    images: Iterable[Mapping[str, Any]], root: Path, workers: int = 16
) -> Counter[str]:
    """Download every photo in parallel; never raises for a dead link."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return Counter(pool.map(lambda m: download_one(m, root), images))


def oriented(raw: Image.Image, ann_width: int, ann_height: int) -> tuple[Image.Image, float]:
    """The version of the photo whose shape matches the annotation, and its scale factor.

    The downloaded file may be the original or a 640 px copy, with or without an EXIF rotation.
    Of the upright and the stored version, pick the one whose aspect ratio matches the annotated
    width x height; boxes are then multiplied by `scale`.
    """
    target = ann_width / ann_height
    candidates = [ImageOps.exif_transpose(raw), raw]
    best = min(candidates, key=lambda im: abs(im.width / im.height - target))
    return best, best.width / ann_width


def crop_box(
    img: Image.Image, bbox: tuple[float, float, float, float], scale: float, pad: float = 0.1
) -> Image.Image:
    """Crop a COCO box (x, y, w, h in annotation pixels) with `pad` margin on each side."""
    x, y, w, h = (v * scale for v in bbox)
    px, py = pad * w, pad * h
    box = (
        max(0, round(x - px)),
        max(0, round(y - py)),
        min(img.width, round(x + w + px)),
        min(img.height, round(y + h + py)),
    )
    return img.crop(box)
