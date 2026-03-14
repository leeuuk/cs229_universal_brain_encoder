"""Voxel-to-Cluster (V2C) utilities for cluster-level ablation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def load_v2c_mapping(
    v2c_dir: str | Path,
    subjects: List[int],
) -> Dict[int, np.ndarray]:
    """Load per-subject cluster IDs (V2C) from directory.

    Expects files: subj01_cluster_ids.npy, subj02_cluster_ids.npy, ...
    Each file is a 1D array of shape (V,) with dtype int64, values in [0, num_clusters-1].

    Parameters
    ----------
    v2c_dir : path to directory containing subjXX_cluster_ids.npy
    subjects : list of subject IDs (e.g. [1, 2, ..., 8])

    Returns
    -------
    dict mapping subject_id -> (V,) int64 array of cluster IDs per voxel
    """
    v2c_dir = Path(v2c_dir)
    out: Dict[int, np.ndarray] = {}
    for sid in subjects:
        path = v2c_dir / f"subj{sid:02d}_cluster_ids.npy"
        if not path.exists():
            raise FileNotFoundError(f"V2C file not found: {path}")
        out[sid] = np.load(path).astype(np.int64)
    return out


def get_keep_mask_for_ablation(
    v2c: np.ndarray,
    exclude_cluster_id: int,
) -> np.ndarray:
    """Boolean mask of voxels to KEEP when ablating (removing) one cluster.

    keep_mask[v] is True iff voxel v does NOT belong to exclude_cluster_id.

    Parameters
    ----------
    v2c : (V,) int array, cluster id per voxel
    exclude_cluster_id : cluster to remove (0-based)

    Returns
    -------
    (V,) bool array, True = keep this voxel
    """
    return (v2c != exclude_cluster_id)


def apply_voxel_mask(
    gt: torch.Tensor,
    pred: torch.Tensor,
    keep_mask: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subset (N,V) tensors to voxels where keep_mask is True.

    Use for retrieval: compute similarity over remaining voxels only.

    Parameters
    ----------
    gt, pred : (N, V) tensors
    keep_mask : (V,) bool array

    Returns
    -------
    gt_masked, pred_masked : (N, V') tensors, V' = keep_mask.sum()
    """
    gt_m = gt[:, keep_mask]
    pred_m = pred[:, keep_mask]
    return gt_m, pred_m


def cluster_voxel_counts(
    v2c_per_subj: Dict[int, np.ndarray],
    num_clusters: int,
) -> Dict[int, np.ndarray]:
    """Per-subject voxel count per cluster.

    Returns
    -------
    dict subject_id -> (num_clusters,) int array of counts
    """
    out: Dict[int, np.ndarray] = {}
    for sid, v2c in v2c_per_subj.items():
        counts = np.bincount(v2c, minlength=num_clusters)
        out[sid] = counts
    return out


def subjects_with_even_cluster_distribution(
    v2c_per_subj: Dict[int, np.ndarray],
    num_clusters: int,
    min_voxels_per_cluster: int = 1,
    max_fraction_empty: float = 0.0,
) -> List[int]:
    """Select subjects where voxels are relatively evenly distributed across clusters.

    Useful for ablation: if a subject has no voxels in cluster c, ablating c has no effect.

    Parameters
    ----------
    v2c_per_subj : from load_v2c_mapping
    num_clusters : K
    min_voxels_per_cluster : require each cluster to have at least this many voxels
    max_fraction_empty : max allowed fraction of clusters with 0 voxels (0 = none empty)

    Returns
    -------
    list of subject IDs that pass the criteria
    """
    selected = []
    for sid, v2c in v2c_per_subj.items():
        counts = np.bincount(v2c, minlength=num_clusters)
        n_empty = (counts == 0).sum()
        if n_empty / num_clusters > max_fraction_empty:
            continue
        if counts.min() < min_voxels_per_cluster:
            continue
        selected.append(sid)
    return sorted(selected)
