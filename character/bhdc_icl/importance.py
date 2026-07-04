from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .tensor_ops import bounded_gain


@dataclass
class ImportanceTensors:
    importance_cog: torch.Tensor
    importance_alloc: torch.Tensor
    moral_gain: torch.Tensor


def compute_importance_tensors(
    density: torch.Tensor,
    cognitive_curvature: torch.Tensor,
    moral_sensitivity: torch.Tensor | None = None,
    max_moral_gain: float = 2.0,
    detach_moral: bool = True,
) -> ImportanceTensors:
    """Two-tensor split from the v18 addendum.

    importance_cog: self-anchored learning-side tensor.
    importance_alloc: per-turn attention/allocation tensor with detached moral gain.

    Persistent learning consumers must use only importance_cog.
    """

    importance_cog = density * cognitive_curvature
    if moral_sensitivity is None:
        moral_gain = torch.ones_like(importance_cog)
    else:
        ms = moral_sensitivity.detach() if detach_moral else moral_sensitivity
        moral_gain = bounded_gain(ms, max_gain=max_moral_gain)
        if moral_gain.shape != importance_cog.shape:
            moral_gain = moral_gain.expand_as(importance_cog)
    importance_alloc = importance_cog * moral_gain
    return ImportanceTensors(
        importance_cog=importance_cog,
        importance_alloc=importance_alloc,
        moral_gain=moral_gain,
    )
