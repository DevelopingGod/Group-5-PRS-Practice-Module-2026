"""Extract reproducible HSV, HOG and LBP features from the WS1 image export.

Run from the repository root:
    uv run --with scikit-image python scripts/ws2_extract_features.py

Outputs are local-only NumPy files aligned to predictions.csv row order.
No scaler is fitted here; scaling belongs inside the training-only ML pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern

ROOT = Path("data/ws2")
IMAGE_SIZE = (128, 128)
LBP_POINTS = 8
LBP_RADIUS = 1


def features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return colour, gradient and texture descriptors as float32 vectors."""
    with Image.open(path) as raw:
        rgb = np.asarray(raw.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32) / 255.0
    hsv = rgb2hsv(rgb)
    colour = np.concatenate(
        [
            np.histogram(hsv[:, :, channel], bins=16, range=(0.0, 1.0), density=False)[0]
            for channel in range(3)
        ]
    ).astype(np.float32)
    colour /= colour.sum()

    gray = rgb2gray(rgb)
    shape = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    # Quantise to 8-bit before LBP to avoid float-comparison instability.
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, LBP_POINTS, LBP_RADIUS, method="uniform")
    texture = np.histogram(lbp, bins=LBP_POINTS + 2, range=(0, LBP_POINTS + 2))[0]
    texture = texture.astype(np.float32)
    texture /= max(float(texture.sum()), 1.0)
    return colour, shape, texture


def extract(predictions: Path, images: Path, output: Path, limit: int | None) -> None:
    """Extract in manifest/prediction order; fail rather than silently skip images."""
    with predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No prediction rows found")

    colour_vectors: list[np.ndarray] = []
    shape_vectors: list[np.ndarray] = []
    texture_vectors: list[np.ndarray] = []
    for index, row in enumerate(rows):
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe image path at row {index}: {relative}")
        path = images / relative
        if not path.is_file():
            raise FileNotFoundError(f"Row {index}: {path}")
        colour, shape, texture = features(path)
        colour_vectors.append(colour)
        shape_vectors.append(shape)
        texture_vectors.append(texture)
        if (index + 1) % 500 == 0:
            print(f"Extracted {index + 1}/{len(rows)} images", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "hsv": np.stack(colour_vectors),
        "hog": np.stack(shape_vectors),
        "lbp": np.stack(texture_vectors),
    }
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array)
    metadata = {
        "n_images": len(rows),
        "image_size": list(IMAGE_SIZE),
        "feature_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "hsv_bins_per_channel": 16,
        "hog_orientations": 9,
        "hog_pixels_per_cell": [16, 16],
        "hog_cells_per_block": [2, 2],
        "lbp_points": LBP_POINTS,
        "lbp_radius": LBP_RADIUS,
        "row_order": "predictions.csv",
    }
    (output / "feature_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv"
    )
    parser.add_argument("--images", type=Path, default=ROOT / "prepared_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "features")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    extract(args.predictions, args.images, args.output, args.limit)


if __name__ == "__main__":
    main()
