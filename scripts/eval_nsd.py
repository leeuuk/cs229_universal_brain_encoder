#!/usr/bin/env python
"""Evaluate a Universal Brain Encoder checkpoint on Algonauts NSD val data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root

from ube.data.algonauts import AlgonautsDataset, collate_algonauts, load_shared_ids
from ube.metrics import pearson_corrcoef, retrieval_accuracy
from ube.models.universal_encoder import UniversalBrainEncoder
from ube.utils.checkpoint import load_checkpoint
from ube.utils.config import load_yaml
from ube.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate checkpoint on Algonauts NSD data.")
    p.add_argument("--config", type=str, default="configs/nsd.yaml")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--split", type=str, default="shared",
                    help="Split to evaluate on (shared|val)")
    p.add_argument("--nsd_expdesign", type=str, default=None,
                    help="Path to nsd_expdesign.mat (required for shared/val split filtering)")
    p.add_argument("--save", type=str, default=None,
                    help="Optional path to save predictions + metrics (.pt)")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root

    set_seed(int(cfg["train"]["seed"]))

    device = resolve_device(cfg["train"].get("device", "auto"))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    # Load shared IDs
    shared_ids = None
    expdesign_path = args.nsd_expdesign or cfg["data"].get("nsd_expdesign", None)
    if expdesign_path is not None:
        shared_ids = load_shared_ids(expdesign_path)

    # Dataset for a single subject
    eval_ds = AlgonautsDataset(
        root=cfg["data"]["root"],
        split=args.split,
        subjects=[args.subject],
        image_size=int(cfg["data"].get("image_size", 224)),
        shared_nsd_ids=shared_ids,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(cfg["eval"].get("batch_images", 8)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        collate_fn=collate_algonauts,
        drop_last=False,
    )

    subject_voxel_sizes = eval_ds.subject_voxel_sizes

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
    ).to(device)

    load_checkpoint(args.ckpt, model=model, optimizer=None, map_location=device, strict=False)
    model.eval()

    chunk = int(cfg["eval"].get("voxel_chunk", 4096))

    all_pred = []
    all_gt = []

    for batch in tqdm(eval_loader, desc=f"eval sub-{args.subject:02d}"):
        images = batch["images"].to(device, non_blocking=True)
        fmri_list = batch["fmri"]

        with torch.cuda.amp.autocast(enabled=use_amp):
            feats = model.extract_features(images)  # (B,L,P,C)

        for i in range(feats.shape[0]):
            gt = fmri_list[i].to(device)
            pred = model.predict_full_from_features(
                feats[i], subject_id=args.subject, chunk_size=chunk,
            )
            if pred.shape != gt.shape:
                raise RuntimeError(f"Pred shape {pred.shape} != GT shape {gt.shape}")
            all_pred.append(pred.detach().cpu())
            all_gt.append(gt.detach().cpu())

    pred_mat = torch.stack(all_pred, dim=0)  # (N, V)
    gt_mat = torch.stack(all_gt, dim=0)      # (N, V)

    corr = pearson_corrcoef(pred_mat, gt_mat, dim=0)  # (V,)
    median = float(corr.median().item())
    p25 = float(corr.kthvalue(int(0.25 * corr.numel()) + 1).values.item())
    p75 = float(corr.kthvalue(int(0.75 * corr.numel()) + 1).values.item())

    retrieval = retrieval_accuracy(gt_mat, pred_mat)

    print(f"Voxel Pearson corr: median={median:.4f} (p25={p25:.4f}, p75={p75:.4f})")
    print(f"Retrieval: Top-1={retrieval.top1:.2f}%  Top-5={retrieval.top5:.2f}%  "
          f"mean-rank={retrieval.mean_rank:.2f}")

    if args.save:
        out = {
            "subject": args.subject,
            "corr": corr,
            "corr_median": median,
            "corr_p25": p25,
            "corr_p75": p75,
            "retrieval": retrieval,
            "pred": pred_mat,
            "gt": gt_mat,
        }
        torch.save(out, args.save)
        print(f"Saved results to {args.save}")


if __name__ == "__main__":
    main()
