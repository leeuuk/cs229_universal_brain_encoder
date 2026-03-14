#!/usr/bin/env python
"""Cluster-level ablation: measure retrieval accuracy when each V2C cluster is removed.

Usage:
    # First run eval_paper.py to get eval_tensors.pt, and build_v2c_gmm.py to get V2C
    python scripts/eval_v2c_ablation.py --eval_dir results/ --v2c_dir data/v2c/

    # Optional: limit subjects (e.g. only those with even cluster distribution)
    python scripts/eval_v2c_ablation.py --eval_dir results/ --v2c_dir data/v2c/ \\
        --subjects 1 2 3 4 5 6 7 8 --num_clusters 128 --out_dir results/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

# Project root so "from ube.xxx" works when run from any cwd
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if (_PROJECT_ROOT / "ube").is_dir():
    sys.path.insert(0, str(_PROJECT_ROOT))
else:
    _cwd = Path.cwd()
    if (_cwd / "ube").is_dir():
        sys.path.insert(0, str(_cwd))
    else:
        sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from ube.metrics import retrieval_accuracy
from ube.v2c import (
    apply_voxel_mask,
    get_keep_mask_for_ablation,
    load_v2c_mapping,
)


def _parse_subjects_from_tensors(tensor_data: Dict[str, Any]) -> List[int]:
    """Infer subject list from eval_tensors.pt keys (S1_gt, S2_gt, ...)."""
    subjects = []
    for k in tensor_data:
        if k.endswith("_gt") and k.startswith("S"):
            try:
                sid = int(k[1 : k.index("_")])
                subjects.append(sid)
            except ValueError:
                pass
    return sorted(subjects)


def run_ablation_one_subject(
    gt: torch.Tensor,
    pred: torch.Tensor,
    v2c: Union[np.ndarray, torch.Tensor],
    num_clusters: int,
) -> Dict[str, Any]:
    """For one subject: baseline retrieval + per-cluster ablated retrieval.

    Returns dict with baseline (top1, top5, mean_rank) and per_cluster list of
    (cluster_id, top1, top5, mean_rank, top1_drop, mean_rank_increase).
    """
    baseline = retrieval_accuracy(gt, pred)
    v2c_np = np.asarray(v2c)

    per_cluster = []
    for c in tqdm(range(num_clusters), desc="clusters", leave=False):
        keep = get_keep_mask_for_ablation(v2c_np, c)  # (V,) bool
        n_keep = keep.sum()
        if n_keep == 0:
            # No voxels left; skip or record NaN
            per_cluster.append({
                "cluster_id": c,
                "top1": float("nan"),
                "top5": float("nan"),
                "mean_rank": float("nan"),
                "top1_drop": float("nan"),
                "mean_rank_increase": float("nan"),
                "n_voxels_kept": 0,
            })
            continue
        gt_abl, pred_abl = apply_voxel_mask(gt, pred, keep)
        ret_abl = retrieval_accuracy(gt_abl, pred_abl)
        top1_drop = baseline.top1 - ret_abl.top1
        mean_rank_inc = ret_abl.mean_rank - baseline.mean_rank
        per_cluster.append({
            "cluster_id": c,
            "top1": ret_abl.top1,
            "top5": ret_abl.top5,
            "mean_rank": ret_abl.mean_rank,
            "top1_drop": top1_drop,
            "mean_rank_increase": mean_rank_inc,
            "n_voxels_kept": int(n_keep),
        })

    return {
        "baseline_top1": baseline.top1,
        "baseline_top5": baseline.top5,
        "baseline_mean_rank": baseline.mean_rank,
        "per_cluster": per_cluster,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="V2C cluster-level ablation: retrieval accuracy when each cluster is removed"
    )
    ap.add_argument(
        "--eval_dir",
        type=str,
        default="results",
        help="Directory containing eval_tensors.pt (from eval_paper.py)",
    )
    ap.add_argument(
        "--v2c_dir",
        type=str,
        required=True,
        help="Directory with subjXX_cluster_ids.npy (from build_v2c_gmm.py)",
    )
    ap.add_argument(
        "--subjects",
        type=int,
        nargs="*",
        default=None,
        help="Subject IDs to include (default: all in eval_tensors.pt)",
    )
    ap.add_argument("--num_clusters", type=int, default=128)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where to save v2c_ablation_results.pt (default: eval_dir)",
    )
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    v2c_dir = Path(args.v2c_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir
    tensor_path = eval_dir / "eval_tensors.pt"

    if not tensor_path.exists():
        raise FileNotFoundError(
            f"eval_tensors.pt not found in {eval_dir}. Run eval_paper.py first."
        )

    print(f"Loading {tensor_path}...")
    tensor_data = torch.load(tensor_path, map_location="cpu", weights_only=False)
    subjects = args.subjects or _parse_subjects_from_tensors(tensor_data)
    if not subjects:
        raise ValueError("No subjects found in eval_tensors.pt (expected keys S1_gt, S2_gt, ...)")

    print(f"Loading V2C from {v2c_dir} for subjects {subjects}...")
    v2c_per_subj = load_v2c_mapping(v2c_dir, subjects)

    num_clusters = args.num_clusters
    results_per_subject: Dict[int, Dict[str, Any]] = {}
    all_top1_drops: Dict[int, List[float]] = {c: [] for c in range(num_clusters)}
    all_mean_rank_inc: Dict[int, List[float]] = {c: [] for c in range(num_clusters)}

    for sid in subjects:
        gt = tensor_data[f"S{sid}_gt"]
        pred = tensor_data[f"S{sid}_pred"]
        if not isinstance(gt, torch.Tensor):
            gt = torch.from_numpy(np.asarray(gt)).float()
            pred = torch.from_numpy(np.asarray(pred)).float()
        v2c = v2c_per_subj[sid]
        if gt.shape[1] != len(v2c):
            raise ValueError(
                f"Subject {sid}: gt voxels {gt.shape[1]} != V2C length {len(v2c)}"
            )

        print(f"  Subject {sid}: running ablation over {num_clusters} clusters...")
        res = run_ablation_one_subject(
            gt, pred, v2c, num_clusters
        )
        results_per_subject[sid] = res
        for item in res["per_cluster"]:
            c = item["cluster_id"]
            if not (item["top1_drop"] != item["top1_drop"]):  # not nan
                all_top1_drops[c].append(item["top1_drop"])
            if not (item["mean_rank_increase"] != item["mean_rank_increase"]):
                all_mean_rank_inc[c].append(item["mean_rank_increase"])

    # Aggregate across subjects (mean per cluster)
    mean_top1_drop = np.array(
        [
            np.nanmean(all_top1_drops[c]) if all_top1_drops[c] else np.nan
            for c in range(num_clusters)
        ]
    )
    mean_rank_increase = np.array(
        [
            np.nanmean(all_mean_rank_inc[c]) if all_mean_rank_inc[c] else np.nan
            for c in range(num_clusters)
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v2c_ablation_results.pt"
    save_dict = {
        "subjects": subjects,
        "num_clusters": num_clusters,
        "results_per_subject": results_per_subject,
        "mean_top1_drop_per_cluster": mean_top1_drop,
        "mean_rank_increase_per_cluster": mean_rank_increase,
        "eval_dir": str(eval_dir),
        "v2c_dir": str(v2c_dir),
    }
    torch.save(save_dict, out_path)
    print(f"\nSaved ablation results to {out_path}")

    # Brief summary
    order = np.argsort(mean_top1_drop)[::-1]
    print("\nTop 10 clusters by mean Top-1 drop (most important for retrieval):")
    for i in range(min(10, num_clusters)):
        c = order[i]
        print(f"  Cluster {c}: mean Top-1 drop = {mean_top1_drop[c]:.2f}%")
    print("\nRun plot_v2c_ablation.py to generate figures.")


if __name__ == "__main__":
    main()
