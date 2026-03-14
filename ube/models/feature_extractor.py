from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .lora import apply_lora_to_dino_attention_out_proj, freeze_module, set_trainable


def _try_load_dinov2(model_name: str) -> nn.Module:
    """Load DINOv2 via torch.hub.
    """
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)  # type: ignore[attr-defined]
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DINOv2 model '{model_name}' via torch.hub. "

            "Make sure you have internet access and a recent PyTorch. "

            f"Original error: {e}"
        )


def _get_embed_dim(backbone: nn.Module) -> int:
    for attr in ("embed_dim", "num_features", "hidden_dim"):
        if hasattr(backbone, attr):
            v = getattr(backbone, attr)
            if isinstance(v, int):
                return v
    # fallback: inspect the last block output dimension?
    raise AttributeError("Could not infer embedding dimension from backbone.")


def _layer_to_block_index(layer_id_1indexed: int) -> int:
    # Paper specifies layers as 1,6,12,18,24 (1-indexed transformer blocks)
    return int(layer_id_1indexed) - 1


class DinoV2MultiLayerFeatures(nn.Module):
    """Extract multi-level patch token features from a DINOv2 ViT backbone."""

    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        layers: Sequence[int] = (1, 6, 12, 18, 24),
        proj_dim: int = 256,
        train_backbone: str = "lora",  # frozen | lora | full
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.layers_1idx = [int(x) for x in layers]
        self.block_indices = [_layer_to_block_index(x) for x in self.layers_1idx]
        self.proj_dim = int(proj_dim)

        self.backbone = _try_load_dinov2(model_name)
        self.embed_dim = _get_embed_dim(self.backbone)

        # Freeze backbone by default; optionally enable LoRA or full finetuning
        train_backbone = train_backbone.lower()
        if train_backbone == "frozen":
            freeze_module(self.backbone)
        elif train_backbone == "lora":
            freeze_module(self.backbone)
            n = apply_lora_to_dino_attention_out_proj(self.backbone, r=lora_rank, alpha=lora_alpha, dropout=0.0)
            if n == 0:
                raise RuntimeError("LoRA requested but no attention output projection layers were modified.")
            # Ensure LoRA params are trainable
            for p in self.backbone.parameters(): 
                if p.requires_grad:
                    continue
            # LoRA layers inserted are trainable by default.
        elif train_backbone == "full":
            set_trainable(self.backbone, True)
        else:
            raise ValueError(f"train_backbone must be one of: frozen|lora|full, got {train_backbone}")

        # Per-layer projection to C (paper: linear projection per layer, then concat)
        self.proj = nn.ModuleList([nn.Linear(self.embed_dim, self.proj_dim) for _ in self.block_indices])

        # Hook cache
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._cache: Dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise AttributeError("Backbone has no .blocks attribute; can't register intermediate hooks.")

        max_idx = max(self.block_indices)
        if max_idx >= len(blocks):
            raise ValueError(f"Requested block index {max_idx} but backbone has only {len(blocks)} blocks.")

        def _make_hook(block_idx: int):
            def hook(_module, _inputs, output):
                # output: (B, 1+P, D)
                self._cache[block_idx] = output
            return hook

        for bidx in self.block_indices:
            h = blocks[bidx].register_forward_hook(_make_hook(bidx))
            self._hook_handles.append(h)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return features of shape (B, L, P, C)."""
        self._cache.clear()
        _ = self.backbone(images)  # triggers hooks

        # Ensure we captured everything
        missing = [b for b in self.block_indices if b not in self._cache]
        if missing:
            raise RuntimeError(f"Missing hooked features for blocks: {missing}")

        feats: List[torch.Tensor] = []
        for proj_layer, bidx in zip(self.proj, self.block_indices):
            tok = self._cache[bidx]  # (B, 1+P, D)
            if tok.ndim != 3:
                raise RuntimeError(f"Unexpected token shape from backbone block {bidx}: {tok.shape}")
            patch_tok = tok[:, 1:, :]  # drop CLS
            feats.append(proj_layer(patch_tok))  # (B, P, C)

        return torch.stack(feats, dim=1)  # (B, L, P, C)

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
