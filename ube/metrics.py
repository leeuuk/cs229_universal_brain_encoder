from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


def pearson_corrcoef(pred: torch.Tensor, target: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation along `dim`.

    If pred/target are (N,V) and dim=0, returns (V,).
    """
    pred = pred.float()
    target = target.float()
    pred = pred - pred.mean(dim=dim, keepdim=True)
    target = target - target.mean(dim=dim, keepdim=True)
    cov = (pred * target).sum(dim=dim)
    pred_var = (pred * pred).sum(dim=dim)
    targ_var = (target * target).sum(dim=dim)
    return cov / (torch.sqrt(pred_var * targ_var) + eps)


@dataclass
class RetrievalMetrics:
    top1: float
    top5: float
    mean_rank: float


def retrieval_accuracy(gt_fmri: torch.Tensor, pred_fmri: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> RetrievalMetrics:
    """Image retrieval metric described in the paper.

    For each query real fMRI vector (gt_fmri[i]), rank all predicted fMRIs (pred_fmri[j])
    by Pearson correlation across voxels, and record the rank of the matching image j=i.

    Args:
        gt_fmri: (N,V) ground truth fMRI vectors
        pred_fmri: (N,V) predicted fMRI vectors (same image order as gt_fmri)
    """
    assert gt_fmri.ndim == 2 and pred_fmri.ndim == 2
    assert gt_fmri.shape == pred_fmri.shape
    N, V = gt_fmri.shape

    # Center across voxels to turn Pearson correlation into cosine similarity of centered vectors.
    gt = gt_fmri.float()
    pr = pred_fmri.float()
    gt = gt - gt.mean(dim=1, keepdim=True)
    pr = pr - pr.mean(dim=1, keepdim=True)

    gt = torch.nn.functional.normalize(gt, dim=1)
    pr = torch.nn.functional.normalize(pr, dim=1)

    # Similarity matrix: (N,N)
    sim = gt @ pr.T

    # rank of correct match
    ranks = []
    for i in range(N):
        # argsort descending
        order = torch.argsort(sim[i], descending=True)
        rank = (order == i).nonzero(as_tuple=False).item() + 1  # 1-indexed
        ranks.append(rank)

    ranks_t = torch.tensor(ranks, dtype=torch.float32)
    top1 = (ranks_t <= 1).float().mean().item() * 100.0
    top5 = (ranks_t <= 5).float().mean().item() * 100.0
    mean_rank = ranks_t.mean().item()
    return RetrievalMetrics(top1=top1, top5=top5, mean_rank=mean_rank)
