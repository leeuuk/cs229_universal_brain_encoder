#!/usr/bin/env python
"""Train the Universal Brain Encoder on Algonauts 2023 NSD data."""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))  # repo root

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ube.data.algonauts import AlgonautsDataset, collate_algonauts, load_shared_ids
from ube.losses import mse_minus_cosine
from ube.models.universal_encoder import UniversalBrainEncoder
from ube.utils.checkpoint import save_checkpoint
from ube.utils.config import load_yaml
from ube.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Universal Brain Encoder on Algonauts NSD data.")
    p.add_argument("--config", type=str, default="configs/nsd.yaml")
    p.add_argument("--data_root", type=str, default=None, help="Override data.root from config")
    p.add_argument("--subjects", type=int, nargs="*", default=None, help="Subject ids (e.g. 1 2 5 7)")
    p.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    p.add_argument("--ckpt_dir", type=str, default=None, help="Override train.ckpt_dir")
    p.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    p.add_argument("--nsd_expdesign", type=str, default=None,
                    help="Path to nsd_expdesign.mat (to exclude shared-1000 images)")
    return p.parse_args()


def sample_voxel_indices(num_voxels: int, k: int, device: torch.device) -> torch.Tensor:
    if k >= num_voxels:
        return torch.arange(num_voxels, device=device)
    return torch.randperm(num_voxels, device=device)[:k]


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.subjects is not None and len(args.subjects) > 0:
        cfg["data"]["subjects"] = list(args.subjects)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.ckpt_dir is not None:
        cfg["train"]["ckpt_dir"] = args.ckpt_dir

    use_wandb = not args.no_wandb and bool(cfg["train"].get("wandb", True))
    if use_wandb:
        wandb.init(
            project=cfg["train"].get("wandb_project", "universal-brain-encoder"),
            name=cfg["train"].get("wandb_run_name", None),
            config=cfg,
        )

    set_seed(int(cfg["train"]["seed"]))

    device = resolve_device(cfg["train"].get("device", "auto"))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    data_root = Path(cfg["data"]["root"])
    subjects = cfg["data"].get("subjects", None)

    # Load shared-1000 IDs for exclusion
    shared_ids = None
    expdesign_path = args.nsd_expdesign or cfg["data"].get("nsd_expdesign", None)
    if expdesign_path is not None:
        shared_ids = load_shared_ids(expdesign_path)
        print(f"Loaded {len(shared_ids)} shared image IDs to exclude from train/val")

    # ---- datasets (Algonauts: already z-scored, no extra normalisation) ----
    train_ds = AlgonautsDataset(
        root=data_root,
        split="train",
        subjects=subjects,
        image_size=int(cfg["data"].get("image_size", 224)),
        shared_nsd_ids=shared_ids,
    )
    val_ds = AlgonautsDataset(
        root=data_root,
        split="val",
        subjects=subjects,
        image_size=int(cfg["data"].get("image_size", 224)),
        shared_nsd_ids=shared_ids,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_images"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        collate_fn=collate_algonauts,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["eval"].get("batch_images", 8)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        collate_fn=collate_algonauts,
        drop_last=False,
    )

    # ---- model ----
    model = UniversalBrainEncoder(
        subject_voxel_sizes=train_ds.subject_voxel_sizes,
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

    # Optimizer over trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            images = batch["images"].to(device, non_blocking=True)
            fmri_list = batch["fmri"]
            subject_ids = batch["subject_ids"]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                feats = model.extract_features(images)  # (B,L,P,C)
                per_image_losses: List[torch.Tensor] = []

                for i in range(feats.shape[0]):
                    sid = int(subject_ids[i])
                    fmri = fmri_list[i]  # (V,)
                    nvox = fmri.shape[0]
                    k = int(cfg["train"]["voxels_per_image"])
                    idx = sample_voxel_indices(nvox, k, device=device)

                    target = fmri.to(device, non_blocking=True)[idx]
                    pred = model.predict_from_features(feats[i], sid, idx)

                    loss_i = mse_minus_cosine(pred, target, alpha=0.1)
                    per_image_losses.append(loss_i)

                loss = torch.stack(per_image_losses).mean()

            scaler.scale(loss).backward()

            grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)

            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % int(cfg["train"].get("log_every", 20)) == 0:
                step_loss = float(loss.detach().cpu().item())
                pbar.set_postfix(loss=step_loss)
                if use_wandb:
                    wandb.log({"train/loss": step_loss, "step": global_step, "epoch": epoch})

        # ---- validation ----
        if epoch % int(cfg["train"].get("val_every", 1)) == 0:
            model.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="val", leave=False):
                    images = batch["images"].to(device, non_blocking=True)
                    fmri_list = batch["fmri"]
                    subject_ids = batch["subject_ids"]

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        feats = model.extract_features(images)
                        per_image_losses = []
                        for i in range(feats.shape[0]):
                            sid = int(subject_ids[i])
                            fmri = fmri_list[i]
                            nvox = fmri.shape[0]
                            k = int(cfg["train"]["voxels_per_image"])
                            idx = sample_voxel_indices(nvox, k, device=device)
                            target = fmri.to(device, non_blocking=True)[idx]
                            pred = model.predict_from_features(feats[i], sid, idx)
                            loss_i = mse_minus_cosine(pred, target, alpha=0.1)
                            if not torch.isfinite(loss_i):
                                print(
                                    f"  [val] NaN/Inf loss: pred has nan={torch.isnan(pred).any().item()} inf={torch.isinf(pred).any().item()}, "
                                    f"target has nan={torch.isnan(target).any().item()} inf={torch.isinf(target).any().item()}"
                                )
                            per_image_losses.append(loss_i)
                        loss = torch.stack(per_image_losses).mean()
                    if torch.isfinite(loss):
                        val_losses.append(float(loss.cpu().item()))
                    else:
                        print(f"  [val] Skipping batch with non-finite loss")
            mean_val = sum(val_losses) / max(1, len(val_losses)) if val_losses else float("nan")
            print(f"epoch {epoch}: val_loss={mean_val:.6f}")
            if use_wandb:
                wandb.log({"val/loss": mean_val, "epoch": epoch, "step": global_step})

            # Best checkpoint (lowest validation loss)
            if mean_val < best_val_loss:
                best_val_loss = mean_val
                extra = {
                    "subject_voxel_sizes": train_ds.subject_voxel_sizes,
                    "subjects": train_ds.subjects,
                }
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model=model, optimizer=optimizer,
                    epoch=epoch, step=global_step,
                    config=cfg, extra=extra,
                )
                print(f"  -> best model saved (val_loss={mean_val:.6f})")
                if use_wandb:
                    wandb.log({"val/best_loss": best_val_loss, "val/best_epoch": epoch})

        # ---- checkpoint ----
        extra = {
            "subject_voxel_sizes": train_ds.subject_voxel_sizes,
            "subjects": train_ds.subjects,
        }
        # last.pt: always save every epoch (latest state)
        save_checkpoint(
            ckpt_dir / "last.pt",
            model=model, optimizer=optimizer,
            epoch=epoch, step=global_step,
            config=cfg, extra=extra,
        )
        # epoch_XXX.pt: only every save_every
        if epoch % int(cfg["train"].get("save_every", 1)) == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt",
                model=model, optimizer=optimizer,
                epoch=epoch, step=global_step,
                config=cfg, extra=extra,
            )

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
