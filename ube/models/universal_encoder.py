from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .cross_attention import VoxelImageCrossAttention
from .feature_extractor import DinoV2MultiLayerFeatures


class SubjectVoxelEmbeddings(nn.Module):
    """A per-subject collection of per-voxel embedding tables."""

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.tables = nn.ModuleDict()

    def has_subject(self, subject_id: int) -> bool:
        return str(int(subject_id)) in self.tables

    def add_subject(self, subject_id: int, num_voxels: int, init_std: float = 0.02) -> None:
        sid = str(int(subject_id))
        if sid in self.tables:
            raise ValueError(f"Subject {subject_id} already exists in voxel embeddings.")
        emb = nn.Embedding(int(num_voxels), self.embedding_dim)
        nn.init.normal_(emb.weight, std=init_std)
        self.tables[sid] = emb

    def num_voxels(self, subject_id: int) -> int:
        sid = str(int(subject_id))
        if sid not in self.tables:
            raise KeyError(f"Unknown subject {subject_id} in voxel embeddings.")
        return self.tables[sid].num_embeddings

    def forward(self, subject_id: int, voxel_indices: torch.Tensor) -> torch.Tensor:
        """Return (N,E) voxel embeddings."""
        sid = str(int(subject_id))
        if sid not in self.tables:
            raise KeyError(f"Unknown subject {subject_id} in voxel embeddings.")
        return self.tables[sid](voxel_indices)


class UniversalBrainEncoder(nn.Module):
    """Universal Brain Encoder (voxel-centric) as described in the paper.

    This module predicts voxel activations for a single image and a set of voxel indices.
    """

    def __init__(
        self,
        subject_voxel_sizes: Dict[int, int],
        image_size: int = 224,
        patch_size: int = 14,
        embedding_dim: int = 256,
        dino_model: str = "dinov2_vitl14",
        dino_layers: Sequence[int] = (1, 6, 12, 18, 24),
        proj_dim: int = 256,
        train_backbone: str = "lora",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        mlp_hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.proj_dim = int(proj_dim)
        self.dino_layers = list(dino_layers)

        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        grid = image_size // patch_size
        num_patches = grid * grid

        # (a) Feature extractor (DINOv2 + per-layer projections)
        self.feature_extractor = DinoV2MultiLayerFeatures(
            model_name=dino_model,
            layers=dino_layers,
            proj_dim=self.proj_dim,
            train_backbone=train_backbone,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # (b) Voxel embeddings (per subject, per voxel)
        self.voxel_embeddings = SubjectVoxelEmbeddings(embedding_dim=self.embedding_dim)
        for subj, nvox in subject_voxel_sizes.items():
            self.voxel_embeddings.add_subject(int(subj), int(nvox))

        # (c) Cross-attention block
        self.cross_attention = VoxelImageCrossAttention(
            num_layers=len(dino_layers),
            num_patches=num_patches,
            proj_dim=self.proj_dim,
            embedding_dim=self.embedding_dim,
            mlp_hidden_mult=mlp_hidden_mult,
        )

    @torch.no_grad()
    def num_voxels(self, subject_id: int) -> int:
        return self.voxel_embeddings.num_voxels(subject_id)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return (B,L,P,C) image features."""
        return self.feature_extractor(images)

    def predict_from_features(
        self,
        image_features_single: torch.Tensor,  # (L,P,C)
        subject_id: int,
        voxel_indices: torch.Tensor,          # (N,)
        return_attention: bool = False,
    ):
        """Predict activation for the given subject voxel indices.

        If return_attention=True, returns (pred, attn_dict) with
        attn_dict['spatial'] (N, L, P), attn_dict['functional'] (N, L).
        """
        voxel_embed = self.voxel_embeddings(subject_id, voxel_indices)  # (N,E)
        out = self.cross_attention(
            image_features_single, voxel_embed, return_attention=return_attention
        )
        return out

    def predict_full_from_features(
        self,
        image_features_single: torch.Tensor,  # (L,P,C)
        subject_id: int,
        chunk_size: int = 4096,
        device: Optional[torch.device] = None,
        return_attention: bool = False,
    ):
        """Predict activation for ALL voxels of a subject in chunks.

        If return_attention=True, returns (pred, attn_dict) where attn_dict
        has 'spatial' (V, L, P) and 'functional' (V, L).
        """
        nvox = self.voxel_embeddings.num_voxels(subject_id)
        dev = device or image_features_single.device
        preds = []
        spatial_list = []
        functional_list = []
        for start in range(0, nvox, chunk_size):
            end = min(nvox, start + chunk_size)
            idx = torch.arange(start, end, device=dev)
            out = self.predict_from_features(
                image_features_single, subject_id, idx, return_attention=return_attention
            )
            if return_attention:
                preds.append(out[0])
                spatial_list.append(out[1]["spatial"])
                functional_list.append(out[1]["functional"])
            else:
                preds.append(out)
        pred_cat = torch.cat(preds, dim=0)
        if not return_attention:
            return pred_cat
        attn_dict = {
            "spatial": torch.cat(spatial_list, dim=0),
            "functional": torch.cat(functional_list, dim=0),
        }
        return pred_cat, attn_dict

    def add_new_subject(self, subject_id: int, num_voxels: int) -> None:
        """Add a new subject embedding table (useful for transfer learning)."""
        self.voxel_embeddings.add_subject(subject_id, num_voxels)
