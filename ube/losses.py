from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_minus_cosine(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    """
    Args:
        pred: (..., N) predictions
        target: (..., N) targets
        alpha: weight for MSE term (paper uses 0.1)
    """
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target)
    cos = F.cosine_similarity(pred, target, dim=-1, eps=eps).mean()
    return alpha * mse - (1.0 - alpha) * cos
