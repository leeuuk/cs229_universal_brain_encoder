#!/usr/bin/env python
"""Plot V2C cluster ablation results: cluster importance for image retrieval.

Usage:
    python scripts/plot_v2c_ablation.py --results results/v2c_ablation_results.pt --out_dir results/vis_v2c
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_results(path: Path) -> Dict[str, Any]:
    """Load v2c_ablation_results.pt."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def plot_sorted_importance(
    mean_top1_drop: np.ndarray,
    out_path: Path,
    num_clusters: int = 128,
    top_k: int = 40,
) -> None:
    """Bar plot: clusters sorted by mean Top-1 drop (highest = most important)."""
    order = np.argsort(mean_top1_drop)[::-1]
    # Show top_k clusters by default to keep readable
    n_show = min(top_k, num_clusters)
    x = np.arange(n_show)
    vals = mean_top1_drop[order[:n_show]]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_show))

    fig, ax = plt.subplots(figsize=(max(8, n_show * 0.2), 4))
    bars = ax.bar(x, vals, color=colors, edgecolor="none")
    ax.set_xlabel("Cluster (ranked by importance)", fontsize=11)
    ax.set_ylabel("Mean Top-1 accuracy drop (%)", fontsize=11)
    ax.set_title("V2C cluster ablation: retrieval importance (top {} of {} clusters)".format(n_show, num_clusters))
    ax.set_xticks(x)
    ax.set_xticklabels([str(order[i]) for i in range(n_show)], fontsize=8, rotation=45)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_subject_cluster_heatmap(
    results_per_subject: Dict[int, Dict[str, Any]],
    subjects: List[int],
    num_clusters: int,
    out_path: Path,
) -> None:
    """Heatmap: subject (rows) × cluster (cols), color = Top-1 drop when that cluster is removed."""
    mat = np.zeros((len(subjects), num_clusters))
    for i, sid in enumerate(subjects):
        res = results_per_subject[sid]
        for item in res["per_cluster"]:
            c = item["cluster_id"]
            d = item["top1_drop"]
            mat[i, c] = d if not np.isnan(d) else 0.0

    fig, ax = plt.subplots(figsize=(14, max(3, len(subjects) * 0.4)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest", vmin=0)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([f"S{s}" for s in subjects])
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Subject")
    ax.set_title("Top-1 accuracy drop when cluster is removed (Subject × Cluster)")
    plt.colorbar(im, ax=ax, label="Top-1 drop (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_cumulative_importance(
    mean_top1_drop: np.ndarray,
    out_path: Path,
) -> None:
    """Cumulative curve: rank clusters by importance, x = rank, y = cumulative Top-1 drop."""
    order = np.argsort(mean_top1_drop)[::-1]
    sorted_drops = mean_top1_drop[order]
    # Replace nan with 0 for cumulative sum
    sorted_drops = np.nan_to_num(sorted_drops, nan=0.0)
    cum = np.cumsum(sorted_drops)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, "b-", linewidth=2)
    ax.set_xlabel("Number of clusters (ranked by importance)", fontsize=11)
    ax.set_ylabel("Cumulative Top-1 accuracy drop (%)", fontsize=11)
    ax.set_title("Cumulative effect of removing top-k most important clusters")
    ax.axhline(y=cum[-1] * 0.5, color="gray", linestyle="--", alpha=0.7, label="50% of total")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_full_128_bars(
    mean_top1_drop: np.ndarray,
    out_path: Path,
    num_clusters: int = 128,
) -> None:
    """Full bar plot of all 128 clusters (thin bars)."""
    order = np.argsort(mean_top1_drop)[::-1]
    x = np.arange(num_clusters)
    vals = mean_top1_drop[order]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x, vals, color="steelblue", edgecolor="none", width=0.8)
    ax.set_xlabel("Cluster (ranked by importance)", fontsize=11)
    ax.set_ylabel("Mean Top-1 accuracy drop (%)", fontsize=11)
    ax.set_title("V2C cluster ablation: all {} clusters".format(num_clusters))
    ax.set_xticks(x[:: max(1, num_clusters // 20)])
    ax.set_xticklabels([str(order[i]) for i in range(0, num_clusters, max(1, num_clusters // 20))])
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot V2C cluster ablation results (importance for retrieval)"
    )
    ap.add_argument(
        "--results",
        type=str,
        default="results/v2c_ablation_results.pt",
        help="Path to v2c_ablation_results.pt",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for figures (default: same as results file dir)",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Number of top clusters to show in sorted bar plot (default: 40)",
    )
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(results_path)
    subjects = data["subjects"]
    num_clusters = data["num_clusters"]
    results_per_subject = data["results_per_subject"]
    mean_top1_drop = data["mean_top1_drop_per_cluster"]
    if isinstance(mean_top1_drop, torch.Tensor):
        mean_top1_drop = mean_top1_drop.numpy()

    print(f"Loaded results: {len(subjects)} subjects, {num_clusters} clusters")

    plot_sorted_importance(
        mean_top1_drop,
        out_dir / "v2c_ablation_sorted_importance.png",
        num_clusters=num_clusters,
        top_k=args.top_k,
    )
    plot_full_128_bars(
        mean_top1_drop,
        out_dir / "v2c_ablation_all_clusters.png",
        num_clusters=num_clusters,
    )
    plot_subject_cluster_heatmap(
        results_per_subject,
        subjects,
        num_clusters,
        out_dir / "v2c_ablation_subject_cluster_heatmap.png",
    )
    plot_cumulative_importance(
        mean_top1_drop,
        out_dir / "v2c_ablation_cumulative.png",
    )

    print(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
