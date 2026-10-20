"""WS2-07: development-time embedding clustering and labelling-priority analysis.

Usage:
  uv run --with scikit-learn --with umap-learn --with hdbscan --with matplotlib \
      python scripts/ws2_cluster_embeddings.py

Fit PCA/UMAP/clustering on TRAIN embeddings only. Validation/test images are
excluded from fitting and labelling priorities to preserve held-out evaluation.
Outputs live under ignored results/ws2/clustering/.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA = Path("data/ws2")
OUT = Path("results/ws2/clustering")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DATA / "v0.2-local-data/predictions.csv")
    parser.add_argument("--embeddings", type=Path, default=DATA / "v0.2-local-data/embeddings.npy")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--max-plot", type=int, default=2500)
    args = parser.parse_args()

    import hdbscan
    import umap

    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    embeddings = np.load(args.embeddings, allow_pickle=False)
    if embeddings.shape != (len(rows), 256):
        raise ValueError(f"Expected ({len(rows)}, 256) embeddings, got {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embeddings")
    train_idx = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    if len(train_idx) == 0:
        raise ValueError("No training images")
    train_rows = [rows[int(i)] for i in train_idx]
    x = embeddings[train_idx].astype(np.float32)
    labels = np.array([row["material"] for row in train_rows])
    rng = np.random.default_rng(SEED)

    # PCA and UMAP are fitted on training data only. UMAP is a visualisation,
    # not the distance space used to score cluster quality or OOD.
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    pca = PCA(n_components=50, random_state=SEED).fit(x_scaled)
    x_pca = pca.transform(x_scaled)
    umap_2d = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="cosine", random_state=SEED, n_jobs=1,
    ).fit_transform(x)

    methods: dict[str, np.ndarray] = {
        "kmeans_10": KMeans(n_clusters=10, n_init=10, random_state=SEED).fit_predict(x_pca),
        "hdbscan": hdbscan.HDBSCAN(
            min_cluster_size=40, min_samples=10, metric="euclidean",
        ).fit_predict(x_pca),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {
        "seed": SEED,
        "fit_split": "train only",
        "n_train": len(train_idx),
        "embedding_dim": 256,
        "pca_components": 50,
        "umap": {"n_neighbors": 30, "min_dist": 0.1, "metric": "cosine"},
        "methods": {},
        "limitations": [
            "Clusters need not correspond to material classes.",
            "UMAP coordinates distort distances and are used only for visualisation.",
            "Labelled metrics describe alignment with known materials, not intrinsic cluster quality.",
            "Image backgrounds and local collection conditions may drive some clusters.",
        ],
    }

    # Inspect representative uncertain training samples; this is NOT an
    # automatic relabelling decision and never draws from validation/test.
    nearest = NearestNeighbors(n_neighbors=2, metric="cosine").fit(x)
    distances, _ = nearest.kneighbors(x)
    novelty = distances[:, 1]
    confidence = np.array([float(row["confidence"]) for row in train_rows])
    ranked = sorted(
        range(len(train_rows)),
        key=lambda i: (confidence[i], -float(novelty[i])),
    )
    with (args.output / "labelling_priority.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "source", "material", "cnn_predicted", "confidence", "nearest_cosine_distance"])
        for i in ranked[:150]:
            row = train_rows[i]
            writer.writerow([
                row["path"], row["source"], row["material"], row["predicted"],
                f"{confidence[i]:.6f}", f"{novelty[i]:.6f}",
            ])

    for name, assignments in methods.items():
        valid = assignments >= 0
        cluster_ids = sorted(set(assignments[valid].tolist()))
        compositions: dict[str, object] = {}
        for cid in cluster_ids:
            members = np.flatnonzero(assignments == cid)
            counts = Counter(labels[members].tolist())
            compositions[str(cid)] = {
                "size": len(members),
                "top_materials": counts.most_common(5),
                "purity": max(counts.values()) / len(members),
                "local_images": sum(train_rows[int(i)]["source"] == "local" for i in members),
            }
        # Silhouette is sampled on the ORIGINAL PCA feature space, not UMAP.
        sample = rng.choice(len(x_pca), size=min(1500, len(x_pca)), replace=False)
        sampled_labels = assignments[sample]
        sampled_valid = sampled_labels >= 0
        silhouette = None
        if 1 < len(set(sampled_labels[sampled_valid])) < int(sampled_valid.sum()):
            silhouette = float(silhouette_score(
                x_pca[sample][sampled_valid],
                sampled_labels[sampled_valid],
            ))
        summaries["methods"][name] = {
            "clusters": len(cluster_ids),
            "noise_fraction": float(np.mean(~valid)),
            "nmi_with_material": float(normalized_mutual_info_score(labels[valid], assignments[valid])) if valid.any() else None,
            "ari_with_material": float(adjusted_rand_score(labels[valid], assignments[valid])) if valid.any() else None,
            "silhouette_sampled_pca": silhouette,
            "composition": compositions,
        }
        fig, ax = plt.subplots(figsize=(10, 7))
        plot_idx = rng.choice(len(train_idx), size=min(args.max_plot, len(train_idx)), replace=False)
        ax.scatter(
            umap_2d[plot_idx, 0], umap_2d[plot_idx, 1],
            c=assignments[plot_idx], s=8, alpha=0.65, cmap="tab20",
            rasterized=True,
        )
        ax.set_title(f"Training embeddings: {name} (UMAP projection)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.tight_layout()
        fig.savefig(args.output / f"{name}_umap.png", dpi=160)
        plt.close(fig)

    # Separate ground-truth view, not used to fit any clustering algorithm.
    classes = sorted(set(labels))
    class_ids = np.array([classes.index(label) for label in labels])
    plot_idx = rng.choice(len(train_idx), size=min(args.max_plot, len(train_idx)), replace=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        umap_2d[plot_idx, 0], umap_2d[plot_idx, 1],
        c=class_ids[plot_idx], s=8, alpha=0.65, cmap="tab10", rasterized=True,
    )
    ax.legend(
        scatter.legend_elements()[0], classes,
        title="Material", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
    )
    ax.set_title("Training embeddings: known material labels (UMAP)")
    fig.tight_layout()
    fig.savefig(args.output / "materials_umap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_train": len(train_idx),
        "methods": {
            name: {key: value for key, value in data.items() if key != "composition"}
            for name, data in summaries["methods"].items()
        },
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
