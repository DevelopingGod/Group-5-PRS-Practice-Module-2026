"""Image sensing: decode and quality-check a photo (WS1).

Thresholds tuned on 13 real phone photos and degraded copies of them (13 Oct 2026); re-check
them on the full local dataset in ml/notebooks/01 (section 4.2):
- Sharpness = variance of the Laplacian on greyscale, measured with the photo's own aspect ratio
  and the long side at most 512 px (never upscaled). Originals scored >= 163; a 2 px Gaussian
  blur scored <= 12; BLUR_MIN = 50 sits between with margin on both sides.
- Motion blur (hand shake) smears only one direction, so edges along it stay sharp and the
  overall Laplacian misses it. Instead, sharpness is measured in 4 directions (horizontal,
  vertical, both diagonals) and the weakest is compared with the strongest. Originals had a
  ratio >= 0.37; every simulated shake (5-15 px, 4 directions) <= 0.25; MOTION_RATIO_MIN = 0.30.
- Too dark: mean grey level < 0.15 (originals 0.37-0.61; rejected from ~30% of normal exposure).
- Too bright: more than 35% of pixels clipped near white (originals <= 10%; 2.2x over-exposure
  median 44%). Mean brightness cannot do this: a white item on a white table is bright but not
  clipped, and must pass.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError

from binbuddy_contracts.api import QualityReport
from binbuddy_contracts.enums import QualityIssue

QUALITY_LONG_SIDE = 512  # px; sharpness depends on scale, so measure at a fixed size
BLUR_MIN = 50.0  # variance of the Laplacian below this = too blurry
DARK_MAX = 0.15
CLIPPED_LEVEL = 250  # grey level counted as blown-out white
CLIPPED_MAX_FRACTION = 0.35
MOTION_RATIO_MIN = 0.30  # weakest / strongest directional sharpness


class InvalidImageError(ValueError):
    pass


def decode_image(image_base64: str) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("could not decode image") from exc
    return img.convert("RGB")


def _grey_for_quality(img: Image.Image) -> npt.NDArray[np.float64]:
    grey = img.convert("L")
    scale = min(1.0, QUALITY_LONG_SIDE / max(grey.size))  # keep aspect ratio, never upscale
    if scale < 1.0:
        size = (max(3, round(grey.width * scale)), max(3, round(grey.height * scale)))
        grey = grey.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(grey, dtype=np.float64)


def motion_ratio(grey: npt.NDArray[np.float64]) -> float:
    """Weakest / strongest directional sharpness (1 = same in every direction).

    Each direction's sharpness is the variance of the second derivative along it. A shaken photo
    keeps edges along the shake but loses edges across it, so the ratio drops.
    """
    g = grey
    c = g[1:-1, 1:-1]
    along = [
        g[1:-1, 2:] - 2 * c + g[1:-1, :-2],  # horizontal
        g[2:, 1:-1] - 2 * c + g[:-2, 1:-1],  # vertical
        g[2:, 2:] - 2 * c + g[:-2, :-2],  # diagonal
        g[2:, :-2] - 2 * c + g[:-2, 2:],  # anti-diagonal
    ]
    var = [float(d.var()) for d in along]
    strongest = max(var)
    return min(var) / strongest if strongest > 1e-9 else 1.0


def check_quality(img: Image.Image) -> QualityReport:
    grey = _grey_for_quality(img)
    lap = (
        -4 * grey[1:-1, 1:-1] + grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
    )
    blur = float(lap.var())
    brightness = float(grey.mean() / 255.0)
    clipped = float((grey >= CLIPPED_LEVEL).mean())
    issues: list[QualityIssue] = []
    if blur < BLUR_MIN or motion_ratio(grey) < MOTION_RATIO_MIN:
        issues.append(QualityIssue.TOO_BLURRY)
    if brightness < DARK_MAX:
        issues.append(QualityIssue.TOO_DARK)
    elif clipped > CLIPPED_MAX_FRACTION:
        issues.append(QualityIssue.TOO_BRIGHT)
    return QualityReport(ok=not issues, blur_score=blur, brightness=brightness, issues=issues)


def average_hash(img: Image.Image) -> str:
    """64-bit average hash as 16 hex chars. WS4 may swap in imagehash.phash (same format)."""
    small = np.asarray(img.convert("L").resize((8, 8)), dtype=np.float64)
    bits = (small > small.mean()).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"
