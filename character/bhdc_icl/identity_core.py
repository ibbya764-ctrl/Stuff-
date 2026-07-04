from __future__ import annotations

import torch

from .types import SelfState
from .tensor_ops import cosine_similarity, l2_normalize


class IdentityCore:
    """Temporal continuity vector.

    This is intentionally simple: EMA identity state plus continuity metric.
    Replace with a recurrent state-space block in a full BHDC model.
    """

    def __init__(self, dim: int = 64, ema_rate: float = 0.03):
        self.dim = dim
        self.ema_rate = ema_rate
        self.state = torch.zeros(dim)
        self.prev_state = torch.zeros(dim)
        self.steps = 0

    def update(self, current_vector: torch.Tensor) -> float:
        current_vector = l2_normalize(current_vector.detach().flatten())
        if current_vector.numel() != self.dim:
            raise ValueError(f"identity vector dim {current_vector.numel()} != {self.dim}")
        self.prev_state = self.state.clone()
        if self.steps == 0:
            self.state = current_vector
        else:
            self.state = l2_normalize((1 - self.ema_rate) * self.state + self.ema_rate * current_vector)
        self.steps += 1
        if self.steps < 2:
            return 0.0
        return float(cosine_similarity(self.prev_state[None, :], self.state[None, :]).item())

    def vector(self) -> torch.Tensor:
        return self.state.clone()
