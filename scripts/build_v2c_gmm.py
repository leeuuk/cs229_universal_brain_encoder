#!/usr/bin/env python
"""Build Voxel-to-Cluster (V2C) mapping via GMM on UBE voxel embeddings.

Usage:
    python scripts/build_v2c_gmm.py --ckpt checkpoints/best.pt --out data/v2c/

    # Optional: limit subjects, change K
    python scripts/build_v2c_gmm.py --ckpt checkpoints/best.pt --out data/v2c/ \\
        --subjects 1 2 3 4 5 6 7 8 --num-clusters 128
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.mixture import GaussianMixture


def extract_voxel_embeddings_from_ckpt(
    ckpt_path: str | Path,
    subjects: List[int],
) -> Dict[int, np.ndarray]:
    """Load raw voxel embedding matrices from UBE checkpoint state_dict.

    Checkpoint must contain model.state_dict() with keys
    voxel_embeddings.tables.{subject_id}.weight -> (num_voxels, embedding_dim).

    Returns
    -------
    dict subject_id -> (V, E) float32 array
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no 'model' state_dict")

    out: Dict[int, np.ndarray] = {}
    for sid in subjects:
        key = f"voxel_embeddings.tables.{sid}.weight"
        if key not in state:
            raise KeyError(
                f"Checkpoint missing {key}. "
                f"Available voxel_embeddings keys: "
                f"{[k for k in state if 'voxel_embeddings' in k]}"
            )
        w = state[key]
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy().astype(np.float32)
        out[sid] = w
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build V2C mapping via GMM on UBE voxel embeddings"
    )
    ap.add_argument("--ckpt", type=str, required=True, help="Path to UBE checkpoint (.pt)")
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for subjXX_cluster_ids.npy",
    )
    ap.add_argument(
        "--subjects",
        nargs="*",
        type=int,
        default=None,
        help="Subject IDs (default: from checkpoint extra.subjects)",
    )
    ap.add_argument("--num-clusters", type=int, default=128)
    ap.add_argument(
        "--covariance-type",
        type=str,
        default="diag",
        choices=["full", "tied", "diag", "spherical"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument(
        "--gmm-out",
        type=str,
        default=None,
        help="Optional path to save fitted GMM (joblib)",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out)

    # Resolve subject list
    if args.subjects is not None and len(args.subjects) > 0:
        subjects = sorted(args.subjects)
    else:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        extra = payload.get("extra", {})
        subj_list = extra.get("subjects", None)
        if subj_list is None:
            raise ValueError(
                "Provide --subjects or ensure checkpoint has extra.subjects"
            )
        subjects = sorted(subj_list)

    # Load embeddings from checkpoint (no full model load)
    subject_embs = extract_voxel_embeddings_from_ckpt(ckpt_path, subjects)
    print(f"Loaded voxel embeddings from {ckpt_path}")
    for s, emb in sorted(subject_embs.items()):
        print(f"  subj{s:02d}: {emb.shape[0]} voxels, dim={emb.shape[1]}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Fit GMM on concatenated embeddings
    ordered = sorted(subject_embs.keys())
    all_emb = np.concatenate([subject_embs[s] for s in ordered], axis=0)
    print(
        f"\nFitting GMM: {all_emb.shape[0]} voxels, dim={all_emb.shape[1]}, "
        f"K={args.num_clusters}"
    )

    gmm = GaussianMixture(
        n_components=args.num_clusters,
        covariance_type=args.covariance_type,
        random_state=args.seed,
        max_iter=args.max_iter,
        verbose=1,
    )
    gmm.fit(all_emb)

    if args.gmm_out:
        import joblib
        Path(args.gmm_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(gmm, args.gmm_out)
        print(f"Saved GMM to {args.gmm_out}")

    # Per-subject cluster assignment
    n = 0
    for s in ordered:
        emb = subject_embs[s]
        labels = gmm.predict(emb).astype(np.int64)
        path = out_dir / f"subj{s:02d}_cluster_ids.npy"
        np.save(path, labels)
        n_uniq = len(np.unique(labels))
        print(f"  subj{s:02d}: saved {path}  (clusters used: {n_uniq}/{args.num_clusters})")
        n += 1

    print(f"\nDone. V2C saved to {out_dir}")


if __name__ == "__main__":
    main()
