"""Image-quality checker (app/quality.py) on synthetic phone-like photos."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from app.quality import MOTION_RATIO_MIN, _grey_for_quality, check_quality, motion_ratio
from binbuddy_contracts.enums import QualityIssue


def _scene(width: int = 1600, height: int = 1200, background: int = 120) -> Image.Image:
    """Textured table with a few sharp-edged objects, like a phone photo of an item."""
    rng = np.random.default_rng(0)
    arr = np.clip(rng.normal(background, 6, size=(height, width, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    d.rectangle([500, 300, 1100, 900], fill=(40, 90, 160))
    d.rectangle([560, 360, 1040, 520], fill=(230, 230, 220))
    d.ellipse([650, 600, 950, 850], fill=(200, 60, 40))
    for i in range(12):
        d.text((580 + 30 * i, 420), "A", fill=(0, 0, 0))
    return img


def test_sharp_well_lit_photo_passes() -> None:
    r = check_quality(_scene())
    assert r.ok, r


@pytest.mark.parametrize("radius", [6, 10])
def test_blurred_photo_is_rejected(radius: int) -> None:
    # radius at full 1600 px resolution; after scaling to 512 px this is a 2-3 px blur
    r = check_quality(_scene().filter(ImageFilter.GaussianBlur(radius)))
    assert QualityIssue.TOO_BLURRY in r.issues


def test_dark_photo_is_rejected() -> None:
    r = check_quality(ImageEnhance.Brightness(_scene()).enhance(0.2))
    assert QualityIssue.TOO_DARK in r.issues


def test_blown_out_photo_is_rejected() -> None:
    r = check_quality(ImageEnhance.Brightness(_scene()).enhance(3.0))
    assert QualityIssue.TOO_BRIGHT in r.issues


def test_white_item_on_white_table_is_not_too_bright() -> None:
    r = check_quality(_scene(background=238))  # bright, but not clipped
    assert QualityIssue.TOO_BRIGHT not in r.issues
    assert r.brightness > 0.8


def test_aspect_ratio_kept_and_small_photos_not_upscaled() -> None:
    assert _grey_for_quality(Image.new("RGB", (2000, 500))).shape == (128, 512)
    assert _grey_for_quality(Image.new("RGB", (300, 200))).shape == (200, 300)


def _shake(img: Image.Image, dy: int, dx: int, length: int) -> Image.Image:
    arr = np.asarray(img, dtype=np.float64)
    out = np.zeros_like(arr)
    for k in range(length):
        s = k - length // 2
        out += np.roll(arr, (s * dy, s * dx), axis=(0, 1))
    return Image.fromarray((out / length).astype(np.uint8))


@pytest.mark.parametrize("direction", [(0, 1), (1, 0), (1, 1), (1, -1)])
def test_shaky_photo_is_rejected(direction: tuple[int, int]) -> None:
    small = _scene().resize((512, 384))  # shake measured at the checker's working scale
    shaken = _shake(small, *direction, length=9)
    assert check_quality(small).ok
    assert QualityIssue.TOO_BLURRY in check_quality(shaken).issues


def test_flat_image_has_no_direction() -> None:
    assert motion_ratio(np.full((50, 50), 128.0)) == 1.0
    assert MOTION_RATIO_MIN < 1.0
