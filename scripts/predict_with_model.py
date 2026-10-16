"""Run the exported WS1 vision model over every image of a manifest (for WS2, WS3 and WS4).

Writes, for each image: the calibrated material probabilities, the predicted material, the
confidence, and the 256-d embedding, exactly as the model API computes them (same code path as
`services/inference/app/vision.py`, without the heatmap). Use it to fit ensembles on validation
predictions, tune thresholds, or study embeddings without retraining anything.

    uv run python scripts/predict_with_model.py \
        --model-dir <folder with config.json, model.onnx, head.npz> \
        --manifest <vision_manifest_v2.csv> --images <prepared_v2 folder> --out <output folder>

Outputs in --out: predictions.csv (one row per image, in manifest order), embeddings.npy
(float32, rows aligned with predictions.csv) and model_info.json (which model produced them).
CPU only; roughly 5-10 images per second on a laptop.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# The API package (services/inference/app) is not installed; put it on the path like its tests do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "inference"))

from app.vision import VisionModel, softmax
from binbuddy_contracts.enums import Material

CLASSES = [m.value for m in Material]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--images", type=Path, required=True, help="folder the manifest paths are relative to"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (quick test)")
    args = ap.parse_args()

    model = VisionModel.load(args.model_dir)
    with args.manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    keep = [c for c in ("path", "source", "split", "material", "group") if rows and c in rows[0]]
    embeddings = np.zeros((len(rows), model.pca_components.shape[0]), dtype=np.float32)
    out: list[dict[str, str]] = []
    t0 = time.time()
    for i, row in enumerate(rows):
        with Image.open(args.images / row["path"]) as img:
            x, _ = model.preprocess(img.convert("RGB"))
        logits, pooled, _ = model.session.run(None, {"image": x})
        probs = softmax(np.asarray(logits[0], dtype=np.float32) / model.temperature)
        embeddings[i] = model.embed(np.asarray(pooled[0], dtype=np.float32))
        rec = {c: row[c] for c in keep}
        rec.update({f"p_{c}": f"{float(v):.6f}" for c, v in zip(CLASSES, probs, strict=True)})
        rec["predicted"] = CLASSES[int(probs.argmax())]
        rec["confidence"] = f"{float(probs.max()):.6f}"
        out.append(rec)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(rows)} images, {(i + 1) / (time.time() - t0):.1f} per second")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    np.save(args.out / "embeddings.npy", embeddings)
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    info = {k: config.get(k) for k in ("model_key", "tag", "run", "data_version", "code_commit")}
    info.update({"temperature": config.get("temperature"), "images": len(out)})
    info["manifest"] = args.manifest.name
    (args.out / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    hits: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in out:
        team = "team photos" if r.get("source") == "local" else "public images"
        hits[(r.get("split", "?"), team)].append(r["predicted"] == r.get("material"))
    print(f"done: {len(out)} images in {time.time() - t0:.0f} s -> {args.out}")
    for (split, team), h in sorted(hits.items()):
        print(f"  {split:5s} {team:13s} accuracy {sum(h) / len(h):.3f} ({len(h)} images)")


if __name__ == "__main__":
    main()
