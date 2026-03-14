from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA-style low-rank adapter for an nn.Linear layer.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        if r <= 0:
            raise ValueError("r must be > 0")

        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Linear(in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, out_features, bias=False)

        # Initialization: A ~ Kaiming, B = 0 (so adapter starts as no-op)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


def _find_attention_out_proj(block: nn.Module) -> Optional[Tuple[nn.Module, str]]:
    """Try common attribute names for the attention output projection."""
    # DINOv2 ViT blocks often have block.attn.proj
    if hasattr(block, "attn"):
        attn = getattr(block, "attn")
        for name in ("proj", "out_proj", "proj_out"):
            if hasattr(attn, name) and isinstance(getattr(attn, name), nn.Linear):
                return attn, name
    return None


def apply_lora_to_dino_attention_out_proj(
    backbone: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> int:
    """Wrap attention output projection with LoRALinear in each transformer block.

    Returns:
        number of layers modified
    """
    modified = 0

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise AttributeError("Backbone has no .blocks attribute; can't apply LoRA automatically.")

    for blk in blocks:
        found = _find_attention_out_proj(blk)
        if found is None:
            continue
        parent, attr = found
        base = getattr(parent, attr)
        if freeze_base:
            for p in base.parameters():
                p.requires_grad = False
        setattr(parent, attr, LoRALinear(base=base, r=r, alpha=alpha, dropout=dropout))
        modified += 1

    return modified


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable
