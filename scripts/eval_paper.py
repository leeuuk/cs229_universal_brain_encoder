#!/usr/bin/env python
"""Evaluate a trained Universal Brain Encoder on the shared-1000 test set.

Usage:
    python scripts/eval_paper.py --ckpt checkpoints/best.pt

    # Override paths:
    python scripts/eval_paper.py --ckpt checkpoints/best.pt \
        --data_root /path/to/algonauts \
        --nsd_expdesign /path/to/nsd_expdesign.mat \
        --out_dir results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ube.data.algonauts import AlgonautsDataset, collate_algonauts, load_shared_ids
from ube.metrics import pearson_corrcoef, retrieval_accuracy
from ube.models.universal_encoder import UniversalBrainEncoder
from ube.utils.checkpoint import load_checkpoint
from ube.utils.config import load_yaml
from ube.utils.seed import resolve_device, set_seed


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper-style evaluation: per-voxel correlation + image retrieval."
    )
    p.add_argument("--config", type=str, default="configs/nsd.yaml")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--data_root", type=str, default=None, help="Override data.root")
    p.add_argument("--nsd_expdesign", type=str, default=None,
                    help="Path to nsd_expdesign.mat")
    p.add_argument("--subjects", type=int, nargs="*", default=None,
                    help="Subject ids (default: all from checkpoint)")
    p.add_argument("--out_dir", type=str, default="results",
                    help="Directory to save figures and metrics JSON")
    p.add_argument("--num_workers", type=int, default=None,
                    help="DataLoader workers (use 0 if torch_shm_manager Permission denied)")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


# ── Per-subject evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_subject(
    model: UniversalBrainEncoder,
    subject_id: int,
    data_root: str | Path,
    shared_ids: set,
    image_size: int,
    batch_size: int,
    num_workers: int,
    chunk_size: int,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, Any]:
    """Run full evaluation for a single subject on the shared split.

    Returns dict with:
        corr       : (V,) per-voxel Pearson correlations
        corr_median: float
        corr_p25   : float
        corr_p75   : float
        top1       : float (%)
        top5       : float (%)
        mean_rank  : float
        pred       : (N, V) predictions
        gt         : (N, V) ground truth
    """
    ds = AlgonautsDataset(
        root=data_root,
        split="shared",
        subjects=[subject_id],
        image_size=image_size,
        shared_nsd_ids=shared_ids,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_algonauts,
        drop_last=False,
    )

    all_pred: List[torch.Tensor] = []
    all_gt: List[torch.Tensor] = []
    all_nsd_ids: List[int] = []

    for batch in tqdm(loader, desc=f"  subj{subject_id:02d}", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        fmri_list = batch["fmri"]
        nsd_ids = batch["nsd_ids"]

        with torch.cuda.amp.autocast(enabled=use_amp):
            feats = model.extract_features(images)  # (B, L, P, C)

        for i in range(feats.shape[0]):
            gt = fmri_list[i].to(device)
            pred = model.predict_full_from_features(
                feats[i], subject_id=subject_id, chunk_size=chunk_size,
            )
            all_pred.append(pred.cpu())
            all_gt.append(gt.cpu())
            all_nsd_ids.append(nsd_ids[i])

    pred_mat = torch.stack(all_pred, dim=0)  # (N, V)
    gt_mat = torch.stack(all_gt, dim=0)      # (N, V)

    # Pearson correlation per voxel (across images, dim=0)
    corr = pearson_corrcoef(pred_mat, gt_mat, dim=0)  # (V,)
    median = float(corr.median().item())
    p25 = float(corr.kthvalue(max(1, int(0.25 * corr.numel()))).values.item())
    p75 = float(corr.kthvalue(max(1, int(0.75 * corr.numel()))).values.item())

    # Image retrieval
    ret = retrieval_accuracy(gt_mat, pred_mat)

    # Determine LH voxel count (for splitting fMRI into LH/RH later)
    # The dataset concatenates [LH, RH], so we can infer from the source data
    key_train = (subject_id, "train")
    key_test = (subject_id, "test")
    if key_train in ds._lh_fmri:
        n_lh = ds._lh_fmri[key_train].shape[1]
    elif key_test in ds._lh_fmri:
        n_lh = ds._lh_fmri[key_test].shape[1]
    else:
        n_lh = None

    return {
        "corr": corr,
        "corr_median": median,
        "corr_p25": p25,
        "corr_p75": p75,
        "top1": ret.top1,
        "top5": ret.top5,
        "mean_rank": ret.mean_rank,
        "pred": pred_mat,
        "gt": gt_mat,
        "nsd_ids": all_nsd_ids,
        "n_images": pred_mat.shape[0],
        "n_voxels": pred_mat.shape[1],
        "n_lh": n_lh,
    }


# ── Plotting (Figure 4 style) ───────────────────────────────────────────────

def plot_results(
    per_subject: Dict[int, Dict[str, Any]],
    out_dir: Path,
) -> None:
    """Generate paper-style evaluation figures (per-subject only)."""
    subjects = sorted(per_subject.keys())
    labels = [f"S{s}" for s in subjects]

    # ── Gather data ──
    corr_data = [per_subject[s]["corr"].numpy() for s in subjects]
    medians = [per_subject[s]["corr_median"] for s in subjects]
    p25s = [per_subject[s]["corr_p25"] for s in subjects]
    p75s = [per_subject[s]["corr_p75"] for s in subjects]
    top1s = [per_subject[s]["top1"] for s in subjects]
    top5s = [per_subject[s]["top5"] for s in subjects]

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── (a) Voxel Correlation Distribution ──
    positions = np.arange(len(labels))
    bp = ax1.boxplot(
        corr_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        whis=[25, 75],
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#4CAF50")
        patch.set_alpha(0.7)

    ax1.scatter(positions, medians, color="red", s=40, zorder=5, label="median")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Pearson Correlation", fontsize=12)
    ax1.set_title("(a) Voxel Correlation Distribution", fontsize=13, fontweight="bold")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ── (b) Retrieval Accuracy ──
    x = np.arange(len(labels))
    bar_w = 0.35
    bars1 = ax2.bar(x - bar_w / 2, top1s, bar_w, label="Top-1", color="#2196F3", alpha=0.85)
    bars2 = ax2.bar(x + bar_w / 2, top5s, bar_w, label="Top-5", color="#2196F3", alpha=0.45,
                    hatch="//", edgecolor="#2196F3")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.8,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("(b) Retrieval Accuracy by Subject", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "eval_paper_figure.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {fig_path}")


# ── Print summary table ──────────────────────────────────────────────────────

def print_summary(per_subject: Dict[int, Dict[str, Any]]) -> None:
    subjects = sorted(per_subject.keys())

    print("\n" + "=" * 78)
    print(f"{'Subject':>8}  {'Images':>6}  {'Voxels':>7}  "
          f"{'Corr med':>9}  {'p25':>6}  {'p75':>6}  "
          f"{'Top-1':>6}  {'Top-5':>6}  {'MeanRk':>7}")
    print("-" * 78)
    for s in subjects:
        r = per_subject[s]
        print(f"{'S' + str(s):>8}  {r['n_images']:>6}  {r['n_voxels']:>7}  "
              f"{r['corr_median']:>9.4f}  {r['corr_p25']:>6.4f}  {r['corr_p75']:>6.4f}  "
              f"{r['top1']:>6.1f}  {r['top5']:>6.1f}  {r['mean_rank']:>7.1f}")
    print("=" * 78)


# ── Main ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root

    set_seed(int(cfg["train"]["seed"]))

    device_str = args.device if args.device != "auto" else cfg["train"].get("device", "auto")
    device = resolve_device(device_str)
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    # ── Shared IDs ──
    expdesign_path = args.nsd_expdesign or cfg["data"].get("nsd_expdesign", None)
    if expdesign_path is None:
        raise ValueError(
            "nsd_expdesign path is required for shared split evaluation. "
            "Provide via --nsd_expdesign or in config data.nsd_expdesign"
        )
    shared_ids = load_shared_ids(expdesign_path)
    print(f"Loaded {len(shared_ids)} shared image IDs")

    # ── Load checkpoint metadata ──
    ckpt_payload = torch.load(args.ckpt, map_location="cpu")
    extra = ckpt_payload.get("extra", {})
    ckpt_subjects = extra.get("subjects", None)
    ckpt_voxel_sizes = extra.get("subject_voxel_sizes", None)

    if args.subjects is not None:
        subjects = sorted(args.subjects)
    elif ckpt_subjects is not None:
        subjects = sorted(ckpt_subjects)
    else:
        raise ValueError("Cannot determine subjects. Provide --subjects or ensure checkpoint has subject info.")

    if ckpt_voxel_sizes is None:
        raise ValueError("Checkpoint missing subject_voxel_sizes in extra metadata.")

    print(f"Subjects to evaluate: {subjects}")
    print(f"Voxel sizes: {ckpt_voxel_sizes}")

    # ── Build model and load weights ──
    model = UniversalBrainEncoder(
        subject_voxel_sizes=ckpt_voxel_sizes,
        image_size=int(cfg["data"].get("image_size", 224)),
        embedding_dim=int(cfg["model"].get("embedding_dim", 256)),
        dino_model=str(cfg["model"].get("dino_model", "dinov2_vitl14")),
        dino_layers=cfg["model"].get("dino_layers", [1, 6, 12, 18, 24]),
        proj_dim=int(cfg["model"].get("proj_dim", 256)),
        train_backbone=str(cfg["model"].get("train_backbone", "lora")),
        lora_rank=int(cfg["model"].get("lora_rank", 8)),
        lora_alpha=int(cfg["model"].get("lora_alpha", 16)),
        mlp_hidden_mult=int(cfg["model"].get("mlp_hidden_mult", 4)),
    ).to(device)

    load_checkpoint(args.ckpt, model=model, optimizer=None, map_location=device, strict=False)
    model.eval()
    print("Model loaded.\n")

    data_root = cfg["data"]["root"]
    image_size = int(cfg["data"].get("image_size", 224))
    batch_size = int(cfg["eval"].get("batch_images", 8))
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["data"].get("num_workers", 4))
    chunk_size = int(cfg["eval"].get("voxel_chunk", 4096))

    # ── Evaluate each subject ──
    per_subject: Dict[int, Dict[str, Any]] = {}
    for sid in subjects:
        print(f"Evaluating subject {sid}...")
        per_subject[sid] = evaluate_subject(
            model=model,
            subject_id=sid,
            data_root=data_root,
            shared_ids=shared_ids,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            chunk_size=chunk_size,
            device=device,
            use_amp=use_amp,
        )

    # ── Print summary ──
    print_summary(per_subject)

    # ── Save outputs ──
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON (exclude large tensors)
    metrics_json: Dict[str, Any] = {}
    for sid in subjects:
        r = per_subject[sid]
        metrics_json[f"S{sid}"] = {
            "n_images": r["n_images"],
            "n_voxels": r["n_voxels"],
            "corr_median": r["corr_median"],
            "corr_p25": r["corr_p25"],
            "corr_p75": r["corr_p75"],
            "top1": r["top1"],
            "top5": r["top5"],
            "mean_rank": r["mean_rank"],
        }
    json_path = out_dir / "eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved: {json_path}")

    # Save full tensors for further analysis (including nsd_ids for visualization)
    tensor_path = out_dir / "eval_tensors.pt"
    tensor_data: Dict[str, Any] = {"data_root": str(data_root)}
    for sid in subjects:
        tensor_data[f"S{sid}_corr"] = per_subject[sid]["corr"]
        tensor_data[f"S{sid}_pred"] = per_subject[sid]["pred"]
        tensor_data[f"S{sid}_gt"] = per_subject[sid]["gt"]
        tensor_data[f"S{sid}_nsd_ids"] = per_subject[sid]["nsd_ids"]
        tensor_data[f"S{sid}_n_lh"] = per_subject[sid].get("n_lh", None)
    torch.save(tensor_data, tensor_path)
    print(f"Tensors saved: {tensor_path}")

    # ── Plot figure ──
    plot_results(per_subject, out_dir)


if __name__ == "__main__":
    main()
