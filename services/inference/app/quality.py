"""Image sensing: decode and quality-check a photo (WS1). Thresholds are PROVISIONAL."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from binbuddy_contracts.api import QualityReport
from binbuddy_contracts.enums import QualityIssue

BLUR_MIN = 60.0  # variance of the Laplacian below this = too blurry
DARK_MAX = 0.15
BRIGHT_MIN = 0.92


class InvalidImageError(ValueError):
    pass


def decode_image(image_base64: str) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("could not decode image") from exc
    return img.convert("RGB")


def check_quality(img: Image.Image) -> QualityReport:
    grey = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float64)
    lap = (
        -4 * grey[1:-1, 1:-1] + grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
    )
    blur = float(lap.var())
    brightness = float(grey.mean() / 255.0)
    issues: list[QualityIssue] = []
    if blur < BLUR_MIN:
        issues.append(QualityIssue.TOO_BLURRY)
    if brightness < DARK_MAX:
        issues.append(QualityIssue.TOO_DARK)
    elif brightness > BRIGHT_MIN:
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
