from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from binbuddy_ml import taco


def test_oriented_picks_matching_shape_and_scale() -> None:
    small_portrait = Image.new("RGB", (480, 640))  # a 640 px copy of a 2448 x 3264 photo
    img, scale = taco.oriented(small_portrait, 2448, 3264)
    assert img.size == (480, 640)
    assert scale == pytest.approx(480 / 2448)


def test_oriented_applies_exif_rotation_when_needed() -> None:
    stored_landscape = Image.new("RGB", (640, 480))
    exif = stored_landscape.getexif()
    exif[0x0112] = 6  # "rotate 90 degrees" orientation tag
    stored_landscape.info["exif"] = exif.tobytes()
    reloaded = _roundtrip(stored_landscape, exif)
    img, scale = taco.oriented(reloaded, 2448, 3264)  # annotation is portrait
    assert img.size == (480, 640)
    assert scale == pytest.approx(480 / 2448)


def _roundtrip(img: Image.Image, exif: Image.Exif) -> Image.Image:
    import io

    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    buf.seek(0)
    return Image.open(buf)


def test_crop_box_scales_and_pads() -> None:
    img = Image.new("RGB", (480, 640))
    crop = taco.crop_box(img, (1000, 1000, 500, 500), scale=480 / 2448)
    side = 500 * 480 / 2448 * 1.2  # 10% padding on each side
    assert crop.size == pytest.approx((side, side), abs=2)


def test_crop_box_is_clipped_to_the_photo() -> None:
    img = Image.new("RGB", (100, 100))
    # box 90..140 padded by 5 on each side = 85..145, clipped to the 100 px photo
    assert taco.crop_box(img, (90, 90, 50, 50), scale=1.0).size == (15, 15)


def test_dead_links_are_counted_not_raised(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    good = b"\xff\xd8\xff" + b"0" * 10
    monkeypatch.setattr(
        taco, "_fetch", lambda url, timeout=30.0: good if url.endswith("_z.jpg") else None
    )
    metas: list[dict[str, Any]] = [
        {
            "file_name": "batch_1/a.jpg",
            "flickr_url": "https://x/a_o.png",
            "flickr_640_url": "https://x/a_z.jpg",
        },
        {"file_name": "batch_1/b.jpg", "flickr_url": "https://x/b_o.png", "flickr_640_url": None},
    ]
    stats = taco.download_all(metas, tmp_path, workers=2)
    assert stats == {"640": 1, "failed": 1}
    assert (tmp_path / "batch_1/a.jpg").read_bytes() == good
    assert taco.download_all(metas[:1], tmp_path) == {"exists": 1}
