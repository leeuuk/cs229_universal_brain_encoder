from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerLayerMLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VoxelImageCrossAttention(nn.Module):
    """Voxel↔Image cross-attention block 

    Inputs:
        image_features: (L, P, C)  multi-layer patch features for ONE image
        voxel_embed:   (N, E)      embeddings for N voxels (of one subject)

    Output:
        pred: (N,) predicted voxel activations for that image
    """

    def __init__(
        self,
        num_layers: int,
        num_patches: int,
        proj_dim: int,
        embedding_dim: int = 256,
        mlp_hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_patches = int(num_patches)
        self.proj_dim = int(proj_dim)
        self.embedding_dim = int(embedding_dim)

        # Positional embedding: P x E (learned)
        self.pos_embed = nn.Parameter(torch.zeros(self.num_patches, self.embedding_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Spatial attention projections
        self.key_proj = nn.Linear(self.proj_dim, self.embedding_dim)
        self.query_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Per-layer MLPs (one per DINO layer)
        self.mlps = nn.ModuleList([PerLayerMLP(self.proj_dim, hidden_mult=mlp_hidden_mult) for _ in range(self.num_layers)])

        # Functional attention keys: (L*C) x E (learned)
        self.functional_keys = nn.Parameter(torch.zeros(self.num_layers * self.proj_dim, self.embedding_dim))
        nn.init.normal_(self.functional_keys, std=0.02)

    def forward(
        self,
        image_features: torch.Tensor,
        voxel_embed: torch.Tensor,
        return_attention: bool = False,
    ):
        if image_features.ndim != 3:
            raise ValueError(f"image_features must be (L,P,C); got {image_features.shape}")
        L, P, C = image_features.shape
        if L != self.num_layers:
            raise ValueError(f"Expected L={self.num_layers} layers but got {L}")
        if P != self.num_patches:
            raise ValueError(f"Expected P={self.num_patches} patches but got {P}")
        if C != self.proj_dim:
            raise ValueError(f"Expected C={self.proj_dim} but got {C}")

        if voxel_embed.ndim != 2 or voxel_embed.shape[1] != self.embedding_dim:
            raise ValueError(f"voxel_embed must be (N,E={self.embedding_dim}); got {voxel_embed.shape}")
        N = voxel_embed.shape[0]

        # Compute keys for each layer (L,P,E)
        keys = self.key_proj(image_features) + self.pos_embed.unsqueeze(0)  # broadcast over L

        # Query for all voxels (N,E)
        q = self.query_proj(voxel_embed)

        # Spatial attention (per layer)
        per_layer_out = []
        spatial_attn_list = []  # (N, P) per layer
        scale = 1.0 / math.sqrt(self.embedding_dim)

        for l in range(L):
            # attn logits: (N,P)
            attn_logits = (q @ keys[l].transpose(0, 1)) * scale
            attn = F.softmax(attn_logits, dim=-1)
            if return_attention:
                spatial_attn_list.append(attn)
            # values: (P,C) -> output: (N,C)
            out = attn @ image_features[l]
            out = self.mlps[l](out)
            per_layer_out.append(out)

        # (N,L,C)
        layer_feat = torch.stack(per_layer_out, dim=1)

        # Functional attention (no scale per paper formula: (qK^T)v^T)
        v = layer_feat.reshape(N, L * C)  # (N, L*C)

        # weights: (N, L*C)
        w = voxel_embed @ self.functional_keys.T

        # pred: (N,)
        pred = (w * v).sum(dim=1)

        if not return_attention:
            return pred

        # Return attention for visualization: spatial (N, L, P), functional (N, L)
        spatial_attn = torch.stack(spatial_attn_list, dim=1)  # (N, L, P)
        functional_w = w.reshape(N, L, C).sum(dim=2)  # (N, L) importance per DINO layer
        return pred, {"spatial": spatial_attn, "functional": functional_w}
