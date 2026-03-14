#!/usr/bin/env python
"""Paper-style visualizations for the Universal Brain Encoder.

Usage:
    python scripts/visualize_paper.py \
        --eval_dir results/ \
        --data_root ../algonauts \
        --subject 1 \
        --out_dir results/vis/

    # With voxel-embedding cluster figures:
    python scripts/visualize_paper.py --eval_dir results/ --data_root ../algonauts \
        --ckpt checkpoints/best.pt --subject 1 --out_dir results/vis/

    # Use training-set images (requires --ckpt first time; then cached in eval_dir/eval_tensors_train.pt):
    python scripts/visualize_paper.py --eval_dir results/ --data_root ../algonauts \
        --ckpt checkpoints/best.pt --image_source train --subject 1 --out_dir results/vis/
    # Force recompute train tensors: add --recompute_train

    # Top images per cluster on ALL images (train+test); cache: eval_tensors_all.pt
    python scripts/visualize_paper.py --eval_dir results/ --data_root ../algonauts \
        --ckpt checkpoints/best.pt --subject 1 --figure_f --out_dir results/vis/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Optional: for voxel-embedding clustering
try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-style visualizations.")
    p.add_argument("--eval_dir", type=str, default="results",
                    help="Directory containing eval_tensors.pt from eval_paper.py")
    p.add_argument("--data_root", type=str, default=None,
                    help="Algonauts data root (containing train_data/, test_data/). "
                         "If not given, reads from eval_tensors.pt metadata.")
    p.add_argument("--subject", type=int, default=1,
                    help="Subject to visualize (default: 1)")
    p.add_argument("--n_examples", type=int, default=4,
                    help="Number of example images for fMRI comparison (default: 4)")
    p.add_argument("--n_retrieval", type=int, default=5,
                    help="Number of retrieval query rows to show (default: 5)")
    p.add_argument("--out_dir", type=str, default=None,
                    help="Output directory (default: eval_dir/vis/)")
    # Image source: shared (eval_tensors.pt) or train (run model on train split)
    p.add_argument("--image_source", type=str, default="shared",
                    choices=["shared", "train"],
                    help="Image set for visualization: 'shared' (default) uses eval_tensors.pt; "
                         "'train' uses training split (requires --ckpt).")
    p.add_argument("--recompute_train", action="store_true",
                    help="When using --image_source train, recompute and overwrite cache (default: use cache if present).")
    p.add_argument("--train_max_samples", type=int, default=10000,
                    help="When using --image_source train, use at most this many images (default: 10000). Use 0 for no limit.")
    # Voxel embedding cluster figures (Fig S14, Fig 7a/S15)
    p.add_argument("--ckpt", type=str, default=None,
                    help="Checkpoint path. If set, run voxel-embedding cluster viz (Fig S14, S15).")
    p.add_argument("--config", type=str, default="configs/nsd.yaml",
                    help="Config YAML for model (used when --ckpt is set).")
    p.add_argument("--n_clusters", type=int, default=20,
                    help="Number of k-means clusters for voxel embeddings (default: 20).")
    p.add_argument("--top_images_per_cluster", type=int, default=6,
                    help="Top-k images to show per cluster (default: 6).")
    p.add_argument("--cluster_smooth", action="store_true",
                    help="Smooth cluster boundaries on brain surface (mode filter) for (d).")
    p.add_argument("--figure_f", action="store_true",
                    help="Also generate (f) top images per cluster using ALL images (train+test); uses eval_tensors_all.pt cache.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Train split tensors (when --image_source train)
# ═══════════════════════════════════════════════════════════════════════════════

def load_train_tensors(
    ckpt_path: Path,
    config_path: Path,
    data_root: Path,
    subject_id: int,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], Optional[int]]:
    """Run model on training split and return pred_mat, gt_mat, nsd_ids, n_lh.

    Uses the same logic as eval_paper.evaluate_subject but with split='train'.
    If max_samples > 0, only the first max_samples images are used (to save time).
    """
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from ube.data.algonauts import AlgonautsDataset, collate_algonauts, load_shared_ids
    from ube.models.universal_encoder import UniversalBrainEncoder
    from ube.utils.checkpoint import load_checkpoint
    from ube.utils.config import load_yaml

    cfg = load_yaml(config_path)
    payload = torch.load(ckpt_path, map_location="cpu")
    extra = payload.get("extra", {})
    subject_voxel_sizes = extra.get("subject_voxel_sizes")
    if subject_voxel_sizes is None:
        raise ValueError("Checkpoint missing subject_voxel_sizes in extra.")

    expdesign_path = cfg["data"].get("nsd_expdesign")
    shared_ids = load_shared_ids(expdesign_path) if expdesign_path else set()

    model = UniversalBrainEncoder(
        subject_voxel_sizes=subject_voxel_sizes,
        image_size=int(cfg["data"].get("image_size", 224)),
        embedding_dim=int(cfg["model"].get("embedding_dim", 256)),
        dino_model=str(cfg["model"].get("dino_model", "dinov2_vitl14")),
        dino_layers=cfg["model"].get("dino_layers", [1, 6, 12, 18, 24]),
        proj_dim=int(cfg["model"].get("proj_dim", 256)),
        train_backbone=str(cfg["model"].get("train_backbone", "lora")),
        lora_rank=int(cfg["model"].get("lora_rank", 8)),
        lora_alpha=int(cfg["model"].get("lora_alpha", 16)),
        mlp_hidden_mult=int(cfg["model"].get("mlp_hidden_mult", 4)),
    )
    load_checkpoint(ckpt_path, model=model, map_location=device, strict=False)
    model.to(device)
    model.eval()

    image_size = int(cfg["data"].get("image_size", 224))
    batch_size = int(cfg["eval"].get("batch_images", 8))
    num_workers = int(cfg["data"].get("num_workers", 4))
    chunk_size = int(cfg["eval"].get("voxel_chunk", 4096))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    full_ds = AlgonautsDataset(
        root=str(data_root),
        split="train",
        subjects=[subject_id],
        image_size=image_size,
        shared_nsd_ids=shared_ids,
    )
    if max_samples and max_samples > 0:
        n_use = min(max_samples, len(full_ds))
        ds = Subset(full_ds, range(n_use))
        print(f"  Using first {n_use} train images (--train_max_samples={max_samples})")
    else:
        ds = full_ds
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_algonauts,
        drop_last=False,
    )

    all_pred: List[torch.Tensor] = []
    all_gt: List[torch.Tensor] = []
    all_nsd_ids: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  train split", leave=False):
            images = batch["images"].to(device, non_blocking=True)
            fmri_list = batch["fmri"]
            nsd_ids = batch["nsd_ids"]

            with torch.cuda.amp.autocast(enabled=use_amp):
                feats = model.extract_features(images)

            for i in range(feats.shape[0]):
                gt = fmri_list[i].to(device)
                pred = model.predict_full_from_features(
                    feats[i], subject_id=subject_id, chunk_size=chunk_size,
                )
                all_pred.append(pred.cpu())
                all_gt.append(gt.cpu())
                all_nsd_ids.append(nsd_ids[i])

    pred_mat = torch.stack(all_pred, dim=0).numpy()
    gt_mat = torch.stack(all_gt, dim=0).numpy()
    key_train = (subject_id, "train")
    n_lh = full_ds._lh_fmri[key_train].shape[1] if key_train in full_ds._lh_fmri else None

    return pred_mat, gt_mat, all_nsd_ids, n_lh


def load_all_tensors(
    ckpt_path: Path,
    config_path: Path,
    data_root: Path,
    subject_id: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[int], Optional[int]]:
    """Run model on ALL images for subject (train + test, no exclusion). Returns pred_mat, gt_mat, nsd_ids, n_lh.
    Used for figure (f). Caller should cache to eval_tensors_all.pt."""
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from ube.data.algonauts import AlgonautsDataset, collate_algonauts, load_shared_ids
    from ube.models.universal_encoder import UniversalBrainEncoder
    from ube.utils.checkpoint import load_checkpoint
    from ube.utils.config import load_yaml

    cfg = load_yaml(config_path)
    payload = torch.load(ckpt_path, map_location="cpu")
    extra = payload.get("extra", {})
    subject_voxel_sizes = extra.get("subject_voxel_sizes")
    if subject_voxel_sizes is None:
        raise ValueError("Checkpoint missing subject_voxel_sizes in extra.")

    # No shared exclusion: use empty set so we get all train + all test
    shared_ids = set()
    model = UniversalBrainEncoder(
        subject_voxel_sizes=subject_voxel_sizes,
        image_size=int(cfg["data"].get("image_size", 224)),
        embedding_dim=int(cfg["model"].get("embedding_dim", 256)),
        dino_model=str(cfg["model"].get("dino_model", "dinov2_vitl14")),
        dino_layers=cfg["model"].get("dino_layers", [1, 6, 12, 18, 24]),
        proj_dim=int(cfg["model"].get("proj_dim", 256)),
        train_backbone=str(cfg["model"].get("train_backbone", "lora")),
        lora_rank=int(cfg["model"].get("lora_rank", 8)),
        lora_alpha=int(cfg["model"].get("lora_alpha", 16)),
        mlp_hidden_mult=int(cfg["model"].get("mlp_hidden_mult", 4)),
    )
    load_checkpoint(ckpt_path, model=model, map_location=device, strict=False)
    model.to(device)
    model.eval()

    image_size = int(cfg["data"].get("image_size", 224))
    batch_size = int(cfg["eval"].get("batch_images", 8))
    num_workers = int(cfg["data"].get("num_workers", 4))
    chunk_size = int(cfg["eval"].get("voxel_chunk", 4096))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    all_pred: List[torch.Tensor] = []
    all_gt: List[torch.Tensor] = []
    all_nsd_ids: List[int] = []
    n_lh: Optional[int] = None

    for split_name, split in [("train", "train"), ("test", "val")]:
        ds = AlgonautsDataset(
            root=str(data_root),
            split=split,
            subjects=[subject_id],
            image_size=image_size,
            shared_nsd_ids=shared_ids,
        )
        if len(ds) == 0:
            continue
        # Dataset uses (subj, "train") or (subj, "test") for val split
        key = (subject_id, "train" if split == "train" else "test")
        if n_lh is None and key in getattr(ds, "_lh_fmri", {}):
            n_lh = ds._lh_fmri[key].shape[1]
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_algonauts,
            drop_last=False,
        )

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  all ({split_name})", leave=False):
                images = batch["images"].to(device, non_blocking=True)
                fmri_list = batch["fmri"]
                nsd_ids = batch["nsd_ids"]
                with torch.cuda.amp.autocast(enabled=use_amp):
                    feats = model.extract_features(images)
                for i in range(feats.shape[0]):
                    gt = fmri_list[i].to(device)
                    pred = model.predict_full_from_features(
                        feats[i], subject_id=subject_id, chunk_size=chunk_size,
                    )
                    all_pred.append(pred.cpu())
                    all_gt.append(gt.cpu())
                    all_nsd_ids.append(nsd_ids[i])

    if not all_pred:
        raise RuntimeError(f"No data for subject {subject_id} (train+test).")
    pred_mat = torch.stack(all_pred, dim=0).numpy()
    gt_mat = torch.stack(all_gt, dim=0).numpy()
    return pred_mat, gt_mat, all_nsd_ids, n_lh


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _find_image_path(data_root: Path, nsd_id: int) -> Optional[Path]:
    """Find the image file for a given NSD ID across all subjects/splits."""
    for split_dir in ["train_data", "test_data"]:
        for subj_dir in sorted((data_root / split_dir).glob("subj*")):
            for img_dir_parent in ["training_split/training_images", "test_split/test_images"]:
                img_dir = subj_dir / img_dir_parent
                if not img_dir.exists():
                    continue
                matches = list(img_dir.glob(f"*_nsd-{nsd_id:05d}.png"))
                if matches:
                    return matches[0]
    return None


def _smooth_surface_labels_mode(stat: np.ndarray, mesh_path, n_iter: int = 2) -> np.ndarray:
    """Smooth discrete labels on a surface mesh using mode filter (neighbors + self). Returns smoothed stat."""
    from collections import Counter, defaultdict
    try:
        import nibabel as nib
        gii = nib.load(str(mesh_path))
        darrays = gii.darrays
        if len(darrays) < 2:
            return stat
        # Second darray is often triangles (F, 3)
        faces = np.asarray(darrays[1].data)
        if faces.ndim != 2 or faces.shape[1] != 3:
            return stat
        n_verts = len(stat)
        # Build adjacency: set of neighbors per vertex
        adj: Dict[int, set] = defaultdict(set)
        for a, b, c in faces:
            i, j, k = int(a), int(b), int(c)
            if i < n_verts and j < n_verts and k < n_verts:
                adj[i].update([j, k])
                adj[j].update([i, k])
                adj[k].update([i, j])
        out = np.array(stat, dtype=float)
        for _ in range(n_iter):
            new_out = np.array(out)
            for v in range(n_verts):
                vals = [out[v]]
                for u in adj.get(v, []):
                    vals.append(out[u])
                vals = [x for x in vals if np.isfinite(x) and not np.isnan(x)]
                if vals:
                    new_out[v] = Counter(vals).most_common(1)[0][0]
            out = new_out
        return out
    except Exception:
        return stat


def _n_vertices_from_surf(surf_path) -> int:
    """Return number of vertices in a surface file (.gii, .gii.gz). Uses nibabel, not np.load."""
    try:
        import nibabel as nib
        mesh = nib.load(str(surf_path))
        return mesh.darrays[0].data.shape[0]
    except Exception:
        # Fallback: nilearn might expose this; fsaverage left has 10242 for 'infl_left'
        return 10242


def _load_image(path: Path, size: int = 224) -> np.ndarray:
    """Load and resize image for display."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return np.array(img)


def _load_roi_masks(
    roi_masks_dir: Path,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load all ROI challenge-space masks and mappings.

    ROIs are **predefined anatomical/functional regions** from the Algonauts 2023
    challenge data (not learned by our model). They come from:
      - roi_masks_dir = data_root/train_data/subjXX/roi_masks/
    File convention (same as Algonauts tutorial):
      - mapping_<class>.npy: dict mapping index -> ROI name (e.g. 1 -> "V1v", 2 -> "V1d")
      - lh.<class>_challenge_space.npy, rh.<class>_challenge_space.npy: per-vertex
        class index (challenge space = visual cortex vertices only).
    ROI classes: prf-visualrois (V1, V2, V3, hV4, …), floc-faces (FFA), floc-places
    (PPA, OPA, RSC), floc-bodies (EBA), floc-words (OWFA, VWFA), streams.

    Returns dict: {roi_name: {"lh": bool_array, "rh": bool_array}}
    """
    roi_classes = [
        "prf-visualrois", "floc-faces", "floc-places",
        "floc-bodies", "floc-words", "streams",
    ]

    rois: Dict[str, Dict[str, np.ndarray]] = {}

    for cls_name in roi_classes:
        mapping_path = roi_masks_dir / f"mapping_{cls_name}.npy"
        if not mapping_path.exists():
            continue
        mapping = np.load(mapping_path, allow_pickle=True).item()

        lh_path = roi_masks_dir / f"lh.{cls_name}_challenge_space.npy"
        rh_path = roi_masks_dir / f"rh.{cls_name}_challenge_space.npy"
        if not lh_path.exists() or not rh_path.exists():
            continue

        lh_class = np.load(lh_path)
        rh_class = np.load(rh_path)

        for idx, name in mapping.items():
            if name == "Unknown" or idx == 0:
                continue
            rois[name] = {
                "lh": (lh_class == idx),
                "rh": (rh_class == idx),
            }

    return rois


# ═══════════════════════════════════════════════════════════════════════════════
# (1) Real vs. Predicted fMRI — brain surface maps
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fmri_comparison(
    pred_mat: np.ndarray,     # (N, V)
    gt_mat: np.ndarray,       # (N, V)
    nsd_ids: List[int],
    n_lh: int,
    data_root: Path,
    roi_masks_dir: Path,
    n_examples: int,
    out_dir: Path,
    name_suffix: str = "",
) -> None:
    """Plot real vs predicted fMRI on brain surface for selected images."""
    print("\n[1/3] Real vs. Predicted fMRI comparison...")

    N, V = gt_mat.shape

    # Pick n_examples images with highest overall correlation (most interesting)
    per_image_corr = []
    for i in range(N):
        c = np.corrcoef(gt_mat[i], pred_mat[i])[0, 1]
        per_image_corr.append(c if np.isfinite(c) else 0.0)
    per_image_corr = np.array(per_image_corr)
    # Mix: pick some high-corr and some mid-corr examples
    sorted_idx = np.argsort(per_image_corr)[::-1]
    top_idx = sorted_idx[:n_examples].tolist()

    # Try nilearn brain surface rendering
    try:
        from nilearn import datasets as ni_datasets
        from nilearn import plotting as ni_plotting

        fsaverage = ni_datasets.fetch_surf_fsaverage("fsaverage")

        lh_fsavg = np.load(roi_masks_dir / "lh.all-vertices_fsaverage_space.npy")
        rh_fsavg = np.load(roi_masks_dir / "rh.all-vertices_fsaverage_space.npy")

        _plot_fmri_brain_surface(
            pred_mat, gt_mat, nsd_ids, n_lh,
            lh_fsavg, rh_fsavg, fsaverage,
            data_root, top_idx, out_dir, name_suffix,
        )
        return
    except ImportError:
        print("  nilearn not available. Using flat voxel plot instead.")
    except Exception as e:
        print(f"  nilearn error: {e}. Using flat voxel plot instead.")

    # Fallback: flat voxel comparison
    _plot_fmri_flat(pred_mat, gt_mat, nsd_ids, data_root, top_idx, out_dir, name_suffix)


def _plot_fmri_brain_surface(
    pred_mat: np.ndarray,
    gt_mat: np.ndarray,
    nsd_ids: List[int],
    n_lh: int,
    lh_fsavg_mask: np.ndarray,
    rh_fsavg_mask: np.ndarray,
    fsaverage: dict,
    data_root: Path,
    example_indices: List[int],
    out_dir: Path,
    name_suffix: str = "",
) -> None:
    """Render real vs predicted fMRI on inflated brain surface (nilearn).

    Algonauts: *h.all-vertices_fsaverage_space.npy are boolean masks (length =
    n_vertices per hemisphere). Challenge-space values go at np.where(mask)[0].
    """
    from nilearn import plotting as ni_plotting

    # Indices where challenge vertices map on the surface (tutorial convention)
    lh_idx = np.where(lh_fsavg_mask)[0]
    rh_idx = np.where(rh_fsavg_mask)[0]

    n_ex = len(example_indices)
    # Global vmax across all examples for consistent colorbar
    all_vals = np.concatenate([gt_mat[example_indices].ravel(), pred_mat[example_indices].ravel()])
    vmax = float(np.percentile(np.abs(all_vals), 98))
    if vmax <= 0:
        vmax = 1.0

    fig = plt.figure(figsize=(4 * n_ex + 0.8, 10))
    gs = gridspec.GridSpec(3, n_ex + 1, figure=fig, hspace=0.05, wspace=0.05, width_ratios=[1] * n_ex + [0.08])

    for col, img_idx in enumerate(example_indices):
        nsd_id = nsd_ids[img_idx]
        gt_vec = gt_mat[img_idx]
        pred_vec = pred_mat[img_idx]

        gt_lh, gt_rh = gt_vec[:n_lh], gt_vec[n_lh:]
        pred_lh, pred_rh = pred_vec[:n_lh], pred_vec[n_lh:]

        # Row 0: stimulus image
        ax_img = fig.add_subplot(gs[0, col])
        img_path = _find_image_path(data_root, nsd_id)
        if img_path is not None:
            ax_img.imshow(_load_image(img_path))
        ax_img.set_title(f"nsd-{nsd_id:05d}", fontsize=9)
        ax_img.axis("off")
        if col == 0:
            ax_img.set_ylabel("Stimulus", fontsize=10, fontweight="bold")

        # Row 1: Real fMRI (GT, left hemisphere, lateral view)
        ax_gt = fig.add_subplot(gs[1, col], projection="3d")
        n_verts_lh = len(lh_fsavg_mask)
        fsavg_gt = np.full(n_verts_lh, np.nan)
        fsavg_gt[lh_idx] = gt_lh
        ni_plotting.plot_surf_stat_map(
            fsaverage["infl_left"], fsavg_gt, hemi="left",
            view="lateral", colorbar=False, vmax=vmax,
            bg_map=fsaverage["sulc_left"], axes=ax_gt,
            cmap="coolwarm", threshold=0.01,
        )
        if col == 0:
            ax_gt.set_ylabel("Real fMRI (GT)", fontsize=10, fontweight="bold")

        # Row 2: Predicted fMRI
        ax_pred = fig.add_subplot(gs[2, col], projection="3d")
        fsavg_pred = np.full_like(fsavg_gt, np.nan)
        fsavg_pred[lh_idx] = pred_lh
        ni_plotting.plot_surf_stat_map(
            fsaverage["infl_left"], fsavg_pred, hemi="left",
            view="lateral", colorbar=False, vmax=vmax,
            bg_map=fsaverage["sulc_left"], axes=ax_pred,
            cmap="coolwarm", threshold=0.01,
        )
        if col == 0:
            ax_pred.set_ylabel("Predicted fMRI", fontsize=10, fontweight="bold")

    # Shared colorbar for fMRI scale (coolwarm: blue = low, red = high)
    cax = fig.add_subplot(gs[1:, n_ex])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, shrink=0.6, label="fMRI activation")
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("(a) Real vs. Predicted fMRI", fontsize=14, fontweight="bold", y=0.98)
    out_path = out_dir / f"vis_fmri_comparison{name_suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_fmri_flat(
    pred_mat: np.ndarray,
    gt_mat: np.ndarray,
    nsd_ids: List[int],
    data_root: Path,
    example_indices: List[int],
    out_dir: Path,
    name_suffix: str = "",
) -> None:
    """Flat (non-surface) comparison of real vs predicted fMRI."""
    n_ex = len(example_indices)
    all_vals = np.concatenate([gt_mat[example_indices].ravel(), pred_mat[example_indices].ravel()])
    vmax = float(np.percentile(np.abs(all_vals), 98))
    if vmax <= 0:
        vmax = 1.0

    fig = plt.figure(figsize=(3.5 * n_ex + 0.6, 8))
    gs = gridspec.GridSpec(3, n_ex + 1, figure=fig, width_ratios=[1] * n_ex + [0.06])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_ex)] for r in range(3)])
    if n_ex == 1:
        axes = axes.reshape(-1, 1)

    for col, img_idx in enumerate(example_indices):
        nsd_id = nsd_ids[img_idx]
        gt_vec = gt_mat[img_idx]
        pred_vec = pred_mat[img_idx]
        corr = np.corrcoef(gt_vec, pred_vec)[0, 1]

        # Row 0: stimulus image
        img_path = _find_image_path(data_root, nsd_id)
        if img_path is not None:
            axes[0, col].imshow(_load_image(img_path))
        axes[0, col].set_title(f"nsd-{nsd_id:05d}", fontsize=9)
        axes[0, col].axis("off")
        if col == 0:
            axes[0, col].set_ylabel("Stimulus", fontsize=10, fontweight="bold")

        # Row 1: real fMRI (GT) as heatmap
        n_side = int(np.ceil(np.sqrt(len(gt_vec))))
        gt_padded = np.full(n_side * n_side, np.nan)
        gt_padded[:len(gt_vec)] = gt_vec
        axes[1, col].imshow(gt_padded.reshape(n_side, n_side),
                            cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].set_ylabel("Real fMRI (GT)", fontsize=10, fontweight="bold")

        # Row 2: predicted fMRI as heatmap
        pred_padded = np.full(n_side * n_side, np.nan)
        pred_padded[:len(pred_vec)] = pred_vec
        im = axes[2, col].imshow(pred_padded.reshape(n_side, n_side),
                                 cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        axes[2, col].axis("off")
        if col == 0:
            axes[2, col].set_ylabel("Predicted fMRI", fontsize=10, fontweight="bold")
        axes[2, col].set_xlabel(f"r={corr:.3f}", fontsize=9)

    cax = fig.add_subplot(gs[1:, n_ex])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, shrink=0.6, label="fMRI activation")

    fig.suptitle("(a) Real vs. Predicted fMRI", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"vis_fmri_comparison{name_suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# (2) Top-5 Retrieved Images
# ═══════════════════════════════════════════════════════════════════════════════

def plot_retrieval_top5(
    pred_mat: np.ndarray,    # (N, V)
    gt_mat: np.ndarray,      # (N, V)
    nsd_ids: List[int],
    data_root: Path,
    n_rows: int,
    out_dir: Path,
    name_suffix: str = "",
) -> None:
    """For selected query fMRIs, show GT image + Top-5 retrieved images."""
    print("\n[2/3] Top-5 Retrieved Images...")

    N = gt_mat.shape[0]

    # Center and normalize for Pearson correlation
    gt = gt_mat - gt_mat.mean(axis=1, keepdims=True)
    pr = pred_mat - pred_mat.mean(axis=1, keepdims=True)
    gt_norm = gt / (np.linalg.norm(gt, axis=1, keepdims=True) + 1e-8)
    pr_norm = pr / (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-8)

    # Similarity matrix: (N, N)
    sim = gt_norm @ pr_norm.T

    # For each query, get rank of correct match and top-5 retrieved
    ranks = []
    top5_indices = []
    for i in range(N):
        order = np.argsort(sim[i])[::-1]
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        top5_indices.append(order[:5].tolist())

    ranks = np.array(ranks)

    # Select diverse query examples:
    # - Some correct (rank=1), some near-miss, some harder
    rank1_idx = np.where(ranks == 1)[0]
    rank_other = np.where(ranks > 1)[0]

    selected = []
    if len(rank1_idx) >= n_rows - 1:
        # Pick a few rank-1 and one harder example
        rng = np.random.RandomState(42)
        selected.extend(rng.choice(rank1_idx, min(n_rows - 1, len(rank1_idx)), replace=False).tolist())
        if len(rank_other) > 0:
            selected.append(rank_other[np.argmin(ranks[rank_other])])
    else:
        selected.extend(rank1_idx.tolist())
        remaining = n_rows - len(selected)
        if len(rank_other) >= remaining:
            sorted_other = rank_other[np.argsort(ranks[rank_other])]
            selected.extend(sorted_other[:remaining].tolist())

    selected = selected[:n_rows]

    # Plot
    n_cols = 6  # GT + 5 retrieved
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Column headers (query = GT image; retrieved = model's top-5 by fMRI similarity)
    col_titles = ["Query (GT)", "Rank-1", "Rank-2", "Rank-3", "Rank-4", "Rank-5"]

    for row, qi in enumerate(selected):
        query_nsd = nsd_ids[qi]
        rank = ranks[qi]
        top5 = top5_indices[qi]

        # GT image
        gt_path = _find_image_path(data_root, query_nsd)
        if gt_path is not None:
            axes[row, 0].imshow(_load_image(gt_path, 200))
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title(col_titles[0], fontsize=10, color="red", fontweight="bold")

        # Top-5 retrieved
        for k in range(5):
            ret_idx = top5[k]
            ret_nsd = nsd_ids[ret_idx]
            ret_path = _find_image_path(data_root, ret_nsd)
            if ret_path is not None:
                axes[row, k + 1].imshow(_load_image(ret_path, 200))

            # Highlight correct match with red border
            if ret_idx == qi:
                for spine in axes[row, k + 1].spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            axes[row, k + 1].axis("off")
            if row == 0:
                axes[row, k + 1].set_title(col_titles[k + 1], fontsize=10)

    fig.suptitle("(b) Top-5 Retrieved Images (by fMRI similarity)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = out_dir / f"vis_retrieval_top5{name_suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# (3) ROI highest-activation images
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roi_top_images(
    pred_mat: np.ndarray,     # (N, V)
    gt_mat: np.ndarray,       # (N, V)
    nsd_ids: List[int],
    n_lh: int,
    data_root: Path,
    roi_masks_dir: Path,
    out_dir: Path,
    top_k: int = 8,
    use_prediction: bool = False,
    name_suffix: str = "",
) -> None:
    """For each major ROI, show images that induce highest mean fMRI activation.

    ROIs are predefined in Algonauts (roi_masks/*.npy). For each ROI we take the
    set of voxels in that region, compute mean fMRI (GT or predicted) over those
    voxels per image, then show the top-k images by that mean activation.
    use_prediction: if True, rank by predicted activation; if False, by GT.
    """
    label = "predicted" if use_prediction else "GT"
    print(f"\n[3/3] ROI highest-activation images ({label})...")

    rois = _load_roi_masks(roi_masks_dir)
    if not rois:
        print("  No ROI masks found. Skipping.")
        return

    # Select representative ROIs (similar to paper Fig. 7)
    target_rois = ["FFA-1", "FFA-2", "EBA", "PPA", "OPA", "RSC",
                   "V1v", "V1d", "hV4", "OWFA", "VWFA-1"]
    available = [r for r in target_rois if r in rois]
    # Also add any other available ROIs not in the target list
    for r in sorted(rois.keys()):
        if r not in available:
            available.append(r)

    if not available:
        print("  No matching ROIs found. Skipping.")
        return

    N = gt_mat.shape[0]
    use_mat = pred_mat if use_prediction else gt_mat

    roi_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for roi_name in available:
        lh_mask = rois[roi_name]["lh"]
        rh_mask = rois[roi_name]["rh"]

        # Build full mask: [LH voxels, RH voxels]
        full_mask = np.concatenate([lh_mask, rh_mask])
        roi_voxels = np.where(full_mask)[0]

        if len(roi_voxels) == 0:
            continue

        # Mean activation across ROI voxels for each image
        mean_act = use_mat[:, roi_voxels].mean(axis=1)  # (N,)
        top_indices = np.argsort(mean_act)[::-1][:top_k]
        roi_results[roi_name] = (top_indices, mean_act[top_indices])

    if not roi_results:
        print("  No ROI voxels found. Skipping.")
        return

    # Limit to a reasonable number of ROIs for display
    display_rois = list(roi_results.keys())[:8]
    n_rois = len(display_rois)

    # Grid: each row is a ROI, columns are top images
    n_cols = min(top_k, 8)
    fig, axes = plt.subplots(n_rois, n_cols, figsize=(2.2 * n_cols, 2.8 * n_rois))
    if n_rois == 1:
        axes = axes.reshape(1, -1)

    for row, roi_name in enumerate(display_rois):
        top_indices, top_acts = roi_results[roi_name]

        for col in range(n_cols):
            if col < len(top_indices):
                img_idx = top_indices[col]
                nsd_id = nsd_ids[img_idx]
                img_path = _find_image_path(data_root, nsd_id)
                if img_path is not None:
                    axes[row, col].imshow(_load_image(img_path, 180))
            axes[row, col].axis("off")

        # ROI label on the left (which region: high = high mean fMRI in those voxels)
        axes[row, 0].text(-0.08, 0.5, roi_name, transform=axes[row, 0].transAxes,
                          fontsize=12, fontweight="bold", va="center", ha="right")

    title_src = "Predicted" if use_prediction else "GT"
    fig.suptitle(f"(c) Images with Highest Activation per Brain ROI (Algonauts ROIs; mean fMRI, {title_src})\n"
                 "(high activation = high mean fMRI in that ROI's voxels)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0.12, 0, 1, 0.96])  # leave left space for ROI labels
    base = "vis_roi_top_images_pred" if use_prediction else "vis_roi_top_images"
    out_path = out_dir / f"{base}{name_suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# (4) Voxel embedding clusters on brain (Fig S14) & (5) Top images per cluster (Fig 7a/S15)
# ═══════════════════════════════════════════════════════════════════════════════

def get_voxel_embedding_cluster_labels(
    ckpt_path: Path,
    config_path: Path,
    subject_id: int,
    n_clusters: int,
) -> np.ndarray:
    """Extract voxel embeddings from the trained model and run k-means clustering.

    Clustering is done on the **voxel embedding vectors** (256-dim per voxel from
    SubjectVoxelEmbeddings), not on fMRI activations. Returns cluster labels 0..n_clusters-1.
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required for voxel embedding clustering. pip install scikit-learn")

    from ube.models.universal_encoder import UniversalBrainEncoder
    from ube.utils.checkpoint import load_checkpoint
    from ube.utils.config import load_yaml

    cfg = load_yaml(config_path)
    payload = torch.load(ckpt_path, map_location="cpu")
    extra = payload.get("extra", {})
    subject_voxel_sizes = extra.get("subject_voxel_sizes")
    if subject_voxel_sizes is None:
        raise ValueError("Checkpoint missing subject_voxel_sizes in extra.")

    model = UniversalBrainEncoder(
        subject_voxel_sizes=subject_voxel_sizes,
        image_size=int(cfg["data"].get("image_size", 224)),
        embedding_dim=int(cfg["model"].get("embedding_dim", 256)),
        dino_model=str(cfg["model"].get("dino_model", "dinov2_vitl14")),
        dino_layers=cfg["model"].get("dino_layers", [1, 6, 12, 18, 24]),
        proj_dim=int(cfg["model"].get("proj_dim", 256)),
        train_backbone=str(cfg["model"].get("train_backbone", "lora")),
        lora_rank=int(cfg["model"].get("lora_rank", 8)),
        lora_alpha=int(cfg["model"].get("lora_alpha", 16)),
        mlp_hidden_mult=int(cfg["model"].get("mlp_hidden_mult", 4)),
    )
    load_checkpoint(ckpt_path, model=model, map_location="cpu", strict=False)
    model.eval()

    sid = str(int(subject_id))
    if sid not in model.voxel_embeddings.tables:
        raise KeyError(f"Subject {subject_id} not in checkpoint.")
    # (V, E) voxel embedding matrix
    emb = model.voxel_embeddings.tables[sid].weight.detach().cpu().numpy()
    V, E = emb.shape

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(emb)  # (V,) 0..n_clusters-1
    return labels


def plot_voxel_embedding_clusters_brain(
    cluster_labels: np.ndarray,
    n_lh: int,
    roi_masks_dir: Path,
    out_dir: Path,
    n_clusters: int = 20,
    name_suffix: str = "",
    smooth: bool = False,
) -> None:
    """Plot brain surface with voxel embedding clusters colored (Fig S14 style).
    If smooth=True, apply mode filter on surface mesh to smooth cluster boundaries."""
    try:
        from nilearn import datasets as ni_datasets
        from nilearn import plotting as ni_plotting
    except ImportError:
        print("  nilearn not available. Skipping voxel cluster brain surface.")
        return

    # Algonauts: *h.all-vertices_fsaverage_space.npy are boolean masks (length = n_vertices per hemisphere)
    lh_fsavg_mask = np.load(roi_masks_dir / "lh.all-vertices_fsaverage_space.npy")
    rh_fsavg_mask = np.load(roi_masks_dir / "rh.all-vertices_fsaverage_space.npy")
    lh_idx = np.where(lh_fsavg_mask)[0]
    rh_idx = np.where(rh_fsavg_mask)[0]
    n_verts_lh = len(lh_fsavg_mask)
    n_verts_rh = len(rh_fsavg_mask)

    fsaverage = ni_datasets.fetch_surf_fsaverage("fsaverage")
    labels_lh = cluster_labels[:n_lh]
    labels_rh = cluster_labels[n_lh:]
    # Map to 1..n_clusters for display (0 = unassigned / background)
    stat_lh = np.full(n_verts_lh, np.nan)
    stat_rh = np.full(n_verts_rh, np.nan)
    stat_lh[lh_idx] = labels_lh + 1
    stat_rh[rh_idx] = labels_rh + 1

    if smooth:
        stat_lh = _smooth_surface_labels_mode(stat_lh, fsaverage["infl_left"], n_iter=2)
        stat_rh = _smooth_surface_labels_mode(stat_rh, fsaverage["infl_right"], n_iter=2)

    # Use GridSpec so colorbar has its own space and does not overlap the brains
    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.02)
    ax_lh = fig.add_subplot(gs[0, 0], projection="3d")
    ax_rh = fig.add_subplot(gs[0, 1], projection="3d")
    cax = fig.add_subplot(gs[0, 2])
    # Discrete colormap for clusters 1..n_clusters
    cmap = plt.cm.get_cmap("tab20", n_clusters)
    vmin, vmax = 1, n_clusters

    ni_plotting.plot_surf_stat_map(
        fsaverage["infl_left"], stat_lh, hemi="left",
        view="lateral", colorbar=False, vmin=vmin, vmax=vmax,
        bg_map=fsaverage["sulc_left"], axes=ax_lh,
        cmap=cmap, threshold=0.5,
    )
    ax_lh.set_title("Left hemisphere", fontsize=11)
    ni_plotting.plot_surf_stat_map(
        fsaverage["infl_right"], stat_rh, hemi="right",
        view="lateral", colorbar=False, vmin=vmin, vmax=vmax,
        bg_map=fsaverage["sulc_right"], axes=ax_rh,
        cmap=cmap, threshold=0.5,
    )
    ax_rh.set_title("Right hemisphere", fontsize=11)

    # Single colorbar on the right (cluster ID 1..n_clusters), no overlap with brains
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, shrink=0.6, label="Cluster ID")
    if n_clusters <= 20:
        cbar.set_ticks(np.arange(1, n_clusters + 1))
        cbar.set_ticklabels(np.arange(1, n_clusters + 1))
    else:
        step = max(1, n_clusters // 10)
        cbar.set_ticks(np.arange(1, n_clusters + 1, step))
        cbar.set_ticklabels(np.arange(1, n_clusters + 1, step))

    fig.suptitle("(d) Voxel Embedding Clusters (k-means on voxel embeddings)" + (" (smoothed)" if smooth else ""), fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    smooth_suffix = "_smooth" if smooth else ""
    out_path = out_dir / f"vis_voxel_embedding_clusters_brain{name_suffix}{smooth_suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cluster_top_images(
    cluster_labels: np.ndarray,
    gt_mat: np.ndarray,
    pred_mat: np.ndarray,
    nsd_ids: List[int],
    data_root: Path,
    out_dir: Path,
    n_clusters: int = 20,
    top_k: int = 6,
    max_display_clusters: int = 20,
    use_prediction: bool = False,
    name_suffix: str = "",
    figure_label: str = "(e)",
) -> None:
    """For each cluster, show images that induce the highest mean fMRI activation (Fig 7a / S15).

    Images come from the data source (shared/train/all). figure_label: "(e)" or "(f)" for title.
    use_prediction: if True, rank by mean predicted fMRI in cluster; if False, by mean GT.
    """
    use_mat = pred_mat if use_prediction else gt_mat
    N, V = use_mat.shape
    n_show = min(n_clusters, max_display_clusters)

    # Per cluster c: voxels_in_c = indices of voxels in this cluster.
    # For each image i, score[i] = mean(use_mat[i, voxels_in_c]) → one number per image.
    cluster_top_indices: Dict[int, np.ndarray] = {}
    for c in range(n_show):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        mean_act = use_mat[:, mask].mean(axis=1)  # (N,) one scalar per image
        top_idx = np.argsort(mean_act)[::-1][:top_k]
        cluster_top_indices[c] = top_idx

    n_rows = n_show
    n_cols = top_k
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_show):
        c = row
        for col in range(n_cols):
            ax = axes[row, col]
            if c in cluster_top_indices and col < len(cluster_top_indices[c]):
                img_idx = cluster_top_indices[c][col]
                nsd_id = nsd_ids[img_idx]
                img_path = _find_image_path(data_root, nsd_id)
                if img_path is not None:
                    ax.imshow(_load_image(img_path, 160))
            ax.axis("off")
        axes[row, 0].set_ylabel(f"Cluster {c + 1}", fontsize=10, fontweight="bold")

    src_label = "Predicted" if use_prediction else "GT"
    fig.suptitle(
        f"{figure_label} Top images per cluster (highest mean fMRI activation in cluster, {src_label})",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    base = "vis_cluster_top_images_pred" if use_prediction else "vis_cluster_top_images"
    out_path = out_dir / f"{base}{name_suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    sid = args.subject

    # ── Load or compute evaluation tensors ──
    if args.image_source == "train":
        cache_path = eval_dir / "eval_tensors_train.pt"
        data_root = Path(args.data_root) if args.data_root else None
        if data_root is None or not data_root.exists():
            raise FileNotFoundError("--image_source train requires a valid --data_root.")

        use_cache = cache_path.exists() and not args.recompute_train
        key_prefix = f"S{sid}"
        requested_max = args.train_max_samples if args.train_max_samples > 0 else None
        if use_cache:
            tensor_data = torch.load(cache_path, map_location="cpu")
            cached_max = tensor_data.get(f"{key_prefix}_train_max_samples")
            if f"{key_prefix}_pred" in tensor_data and cached_max == requested_max:
                pred_mat = tensor_data[f"{key_prefix}_pred"].numpy()
                gt_mat = tensor_data[f"{key_prefix}_gt"].numpy()
                nsd_ids = tensor_data[f"{key_prefix}_nsd_ids"]
                n_lh = tensor_data.get(f"{key_prefix}_n_lh", None)
                if tensor_data.get("data_root"):
                    data_root = Path(tensor_data["data_root"])
                print(f"Loaded train tensors from cache: {cache_path} (subject {sid}, n={len(nsd_ids)})")
            else:
                use_cache = False
        if not use_cache:
            if args.ckpt is None:
                raise ValueError("--image_source train requires --ckpt (needed to compute train tensors).")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Computing tensors from training split (subject {sid})...")
            max_samples = args.train_max_samples if args.train_max_samples > 0 else None
            pred_mat, gt_mat, nsd_ids, n_lh = load_train_tensors(
                ckpt_path=Path(args.ckpt),
                config_path=Path(args.config),
                data_root=data_root,
                subject_id=sid,
                device=device,
                max_samples=max_samples,
            )
            nsd_ids = list(nsd_ids)
            # Save to cache for next time
            existing = {}
            if cache_path.exists():
                existing = torch.load(cache_path, map_location="cpu")
                if not isinstance(existing, dict):
                    existing = {}
            existing["data_root"] = str(data_root)
            existing[f"{key_prefix}_pred"] = torch.from_numpy(pred_mat)
            existing[f"{key_prefix}_gt"] = torch.from_numpy(gt_mat)
            existing[f"{key_prefix}_nsd_ids"] = nsd_ids
            existing[f"{key_prefix}_n_lh"] = n_lh
            existing[f"{key_prefix}_train_max_samples"] = max_samples
            torch.save(existing, cache_path)
            print(f"  Saved train tensors to {cache_path}")
    else:
        tensor_path = eval_dir / "eval_tensors.pt"
        if not tensor_path.exists():
            raise FileNotFoundError(
                f"eval_tensors.pt not found in {eval_dir}. "
                "Run eval_paper.py first."
            )
        tensor_data = torch.load(tensor_path, map_location="cpu")

        key_prefix = f"S{sid}"
        if f"{key_prefix}_pred" not in tensor_data:
            available = [k.split("_")[0] for k in tensor_data if k.endswith("_pred")]
            raise ValueError(
                f"Subject {sid} not found in eval_tensors.pt. "
                f"Available: {sorted(set(available))}"
            )

        pred_mat = tensor_data[f"{key_prefix}_pred"].numpy()  # (N, V)
        gt_mat = tensor_data[f"{key_prefix}_gt"].numpy()      # (N, V)
        nsd_ids = tensor_data[f"{key_prefix}_nsd_ids"]
        n_lh = tensor_data.get(f"{key_prefix}_n_lh", None)
        data_root = Path(args.data_root) if args.data_root else Path(tensor_data.get("data_root", ""))

    print(f"Subject {sid}: {pred_mat.shape[0]} images ({args.image_source}), {pred_mat.shape[1]} voxels")
    if n_lh is not None:
        print(f"  LH voxels: {n_lh}, RH voxels: {pred_mat.shape[1] - n_lh}")

    # ── Resolve paths ──
    if args.image_source == "shared" and (data_root is None or not data_root.exists()):
        raise FileNotFoundError(f"Data root not found: {data_root}. Provide --data_root.")

    roi_masks_dir = data_root / "train_data" / f"subj{sid:02d}" / "roi_masks"
    if not roi_masks_dir.exists():
        print(f"Warning: ROI masks not found at {roi_masks_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    name_suffix = f"_S{sid}_{args.image_source}"  # e.g. _S1_shared, _S1_train

    # ── (1) Real vs Predicted fMRI ──
    plot_fmri_comparison(
        pred_mat=pred_mat,
        gt_mat=gt_mat,
        nsd_ids=nsd_ids,
        n_lh=n_lh,
        data_root=data_root,
        roi_masks_dir=roi_masks_dir,
        n_examples=args.n_examples,
        out_dir=out_dir,
        name_suffix=name_suffix,
    )

    # ── (2) Top-5 Retrieval ──
    plot_retrieval_top5(
        pred_mat=pred_mat,
        gt_mat=gt_mat,
        nsd_ids=nsd_ids,
        data_root=data_root,
        n_rows=args.n_retrieval,
        out_dir=out_dir,
        name_suffix=name_suffix,
    )

    # ── (3) ROI highest activation ──
    if roi_masks_dir.exists() and n_lh is not None:
        plot_roi_top_images(
            pred_mat=pred_mat,
            gt_mat=gt_mat,
            nsd_ids=nsd_ids,
            n_lh=n_lh,
            data_root=data_root,
            roi_masks_dir=roi_masks_dir,
            out_dir=out_dir,
            use_prediction=False,
            name_suffix=name_suffix,
        )
        plot_roi_top_images(
            pred_mat=pred_mat,
            gt_mat=gt_mat,
            nsd_ids=nsd_ids,
            n_lh=n_lh,
            data_root=data_root,
            roi_masks_dir=roi_masks_dir,
            out_dir=out_dir,
            use_prediction=True,
            name_suffix=name_suffix,
        )
    else:
        print("\n[3/3] Skipping ROI visualization (missing roi_masks or n_lh).")

    # ── (4) & (5) Voxel embedding clusters (Fig S14, Fig 7a/S15) ──
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            print(f"\n[4/5] Skipping cluster viz: checkpoint not found: {ckpt_path}")
        elif not _HAS_SKLEARN:
            print("\n[4/5] Skipping cluster viz: scikit-learn not installed.")
        else:
            print("\n[4/5] Voxel embedding clusters (k-means on voxel embeddings)...")
            cluster_labels = get_voxel_embedding_cluster_labels(
                ckpt_path=ckpt_path,
                config_path=Path(args.config),
                subject_id=sid,
                n_clusters=args.n_clusters,
            )
            if roi_masks_dir.exists() and n_lh is not None:
                plot_voxel_embedding_clusters_brain(
                    cluster_labels=cluster_labels,
                    n_lh=n_lh,
                    roi_masks_dir=roi_masks_dir,
                    out_dir=out_dir,
                    n_clusters=args.n_clusters,
                    name_suffix=name_suffix,
                    smooth=args.cluster_smooth,
                )
            else:
                print("  Skipping cluster brain surface (missing roi_masks or n_lh).")
            plot_cluster_top_images(
                cluster_labels=cluster_labels,
                gt_mat=gt_mat,
                pred_mat=pred_mat,
                nsd_ids=nsd_ids,
                data_root=data_root,
                out_dir=out_dir,
                n_clusters=args.n_clusters,
                top_k=args.top_images_per_cluster,
                use_prediction=False,
                name_suffix=name_suffix,
            )
            plot_cluster_top_images(
                cluster_labels=cluster_labels,
                gt_mat=gt_mat,
                pred_mat=pred_mat,
                nsd_ids=nsd_ids,
                data_root=data_root,
                out_dir=out_dir,
                n_clusters=args.n_clusters,
                top_k=args.top_images_per_cluster,
                use_prediction=True,
                name_suffix=name_suffix,
            )

            # ── (f) Top images per cluster using ALL images (train+test) ──
            if args.figure_f:
                all_cache_path = eval_dir / "eval_tensors_all.pt"
                key_prefix = f"S{sid}"
                data_root_f = Path(args.data_root) if args.data_root else None
                have_all_data = False
                if data_root_f is None or not data_root_f.exists():
                    print("  Skipping figure (f): --data_root required.")
                else:
                    if all_cache_path.exists():
                        try:
                            all_data = torch.load(all_cache_path, map_location="cpu")
                            if f"{key_prefix}_pred" in all_data:
                                pred_all = all_data[f"{key_prefix}_pred"].numpy()
                                gt_all = all_data[f"{key_prefix}_gt"].numpy()
                                nsd_ids_all = all_data[f"{key_prefix}_nsd_ids"]
                                n_lh_all = all_data.get(f"{key_prefix}_n_lh")
                                have_all_data = True
                                print(f"  Loaded all-image tensors from cache: {all_cache_path} (n={len(nsd_ids_all)})")
                        except Exception:
                            pass
                    if not have_all_data:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        print("  Computing tensors for ALL images (train+test)...")
                        pred_all, gt_all, nsd_ids_all, n_lh_all = load_all_tensors(
                            ckpt_path=ckpt_path,
                            config_path=Path(args.config),
                            data_root=data_root_f,
                            subject_id=sid,
                            device=device,
                        )
                        nsd_ids_all = list(nsd_ids_all)
                        have_all_data = True
                        existing = {}
                        if all_cache_path.exists():
                            try:
                                existing = torch.load(all_cache_path, map_location="cpu")
                                if not isinstance(existing, dict):
                                    existing = {}
                            except Exception:
                                pass
                        existing["data_root"] = str(data_root_f)
                        existing[f"{key_prefix}_pred"] = torch.from_numpy(pred_all)
                        existing[f"{key_prefix}_gt"] = torch.from_numpy(gt_all)
                        existing[f"{key_prefix}_nsd_ids"] = nsd_ids_all
                        existing[f"{key_prefix}_n_lh"] = n_lh_all
                        torch.save(existing, all_cache_path)
                        print(f"  Saved all-image tensors to {all_cache_path}")
                    if have_all_data:
                        name_suffix_f = f"_S{sid}_all"
                        plot_cluster_top_images(
                            cluster_labels=cluster_labels,
                            gt_mat=gt_all,
                            pred_mat=pred_all,
                            nsd_ids=nsd_ids_all,
                            data_root=data_root_f,
                            out_dir=out_dir,
                            n_clusters=args.n_clusters,
                            top_k=args.top_images_per_cluster,
                            use_prediction=False,
                            name_suffix=name_suffix_f,
                            figure_label="(f)",
                        )
                        plot_cluster_top_images(
                            cluster_labels=cluster_labels,
                            gt_mat=gt_all,
                            pred_mat=pred_all,
                            nsd_ids=nsd_ids_all,
                            data_root=data_root_f,
                            out_dir=out_dir,
                            n_clusters=args.n_clusters,
                            top_k=args.top_images_per_cluster,
                            use_prediction=True,
                            name_suffix=name_suffix_f,
                            figure_label="(f)",
                        )

    print(f"\nAll visualizations saved to: {out_dir}")


if __name__ == "__main__":
    main()
