"""WS2-07 extension: visualise whether CNN embeddings group by dataset source.

Run:
  uv run --with umap-learn --with matplotlib python scripts/ws2_umap_sources.py

Fit UMAP on training embeddings only. The output is exploratory, not a
quantitative proof of domain shift or a model-selection result.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("data/ws2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=ROOT / "v0.2-local-data/predictions.csv"
    )
    parser.add_argument("--embeddings", type=Path, default=ROOT / "v0.2-local-data/embeddings.npy")
    parser.add_argument("--output", type=Path, default=Path("results/ws2/clustering"))
    parser.add_argument("--max-plot", type=int, default=4000)
    args = parser.parse_args()

    import umap

    with args.predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    embeddings = np.load(args.embeddings, allow_pickle=False, mmap_mode="r")
    if embeddings.shape != (len(rows), 256):
        raise ValueError("Embedding and CSV rows do not align")
    train = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    if not len(train):
        raise ValueError("No training rows")
    x = np.asarray(embeddings[train], dtype=np.float32)
    sources = np.array([rows[int(i)]["source"] for i in train])
    materials = np.array([rows[int(i)]["material"] for i in train])
    rng = np.random.default_rng(42)
    sample = rng.choice(len(train), size=min(args.max_plot, len(train)), replace=False)
    coords = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    ).fit_transform(x)
    unique = sorted(set(sources.tolist()))
    palette = plt.get_cmap("tab10")
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    for j, source in enumerate(unique):
        selected = sample[sources[sample] == source]
        ax.scatter(
            coords[selected, 0],
            coords[selected, 1],
            s=9,
            alpha=0.65,
            color=palette(j),
            label=f"{source} (n={sum(sources == source)})",
            rasterized=True,
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title("ConvNeXt training embeddings: dataset source (UMAP)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(args.output / "sources_umap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Quantitative context: class proportions differ between datasets.
    cross = {
        source: dict(sorted(Counter(materials[sources == source].tolist()).items()))
        for source in unique
    }
    report = {
        "fit_split": "train",
        "n_train": len(train),
        "source_counts": dict(sorted(Counter(sources.tolist()).items())),
        "source_material_counts": cross,
        "umap_parameters": {"metric": "cosine", "n_neighbors": 30, "min_dist": 0.1, "seed": 42},
        "limitations": [
            "UMAP axes and apparent gaps are not calibrated distances.",
            "Different source class proportions can create apparent source separation.",
            "This plot cannot establish that shop backgrounds caused errors.",
        ],
    }
    (args.output / "sources_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_train": len(train),
                "source_counts": report["source_counts"],
                "plot": str(args.output / "sources_umap.png"),
                "summary": str(args.output / "sources_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
