from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return l2_normalize(a) @ l2_normalize(b).T


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1, eps=eps)


def pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine distance for rows of x."""
    if x.shape[0] < 2:
        return x.new_tensor(0.0)
    sim = cosine_matrix(x, x)
    n = x.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return (1.0 - sim[mask]).mean()


def bounded_gain(x: torch.Tensor, max_gain: float = 2.0) -> torch.Tensor:
    """Non-negative, bounded moral attention gain.

    This is intentionally detached by callers before persistent learning. The
    returned value is multiplicative: 1 means no extra allocation.

    Recentred so NEUTRAL sensitivity (x=0) yields unit gain (1.0). The old form
    ``1 + max_gain*sigmoid(x)`` mapped x=0 to 2.0, so on a conserved/normalized
    budget only the *variation* of the moral signal carried information and the
    sigmoid compressed everything around 2.0. Here x=0 -> 1.0, large x ->
    1+max_gain, negative x -> 1.0 (clamped): gain only ever adds allocation.
    """
    return 1.0 + max_gain * (2.0 * torch.sigmoid(x) - 1.0).clamp(min=0.0)


def safe_scalar(x: object, default: float = 0.0) -> float:
    try:
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return default
