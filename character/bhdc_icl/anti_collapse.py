from __future__ import annotations

from dataclasses import dataclass

import torch

from .tensor_ops import pairwise_cosine_distance


@dataclass
class AntiCollapseReport:
    spread: float
    target_spread: float
    loss: torch.Tensor
    collapsed: bool


class MoralAntiCollapse:
    """Keep a conscience committee from becoming an approval echo.

    The target is moderate diversity, not maximum disagreement.
    """

    def __init__(self, target_spread: float = 0.25, tolerance: float = 0.08, min_spread: float = 0.05):
        self.target_spread = target_spread
        self.tolerance = tolerance
        self.min_spread = min_spread

    def __call__(self, frame_embeddings: torch.Tensor) -> AntiCollapseReport:
        spread_t = pairwise_cosine_distance(frame_embeddings)
        loss = (spread_t - self.target_spread).pow(2)
        spread = float(spread_t.detach().cpu().item())
        return AntiCollapseReport(
            spread=spread,
            target_spread=self.target_spread,
            loss=loss,
            collapsed=spread < self.min_spread,
        )


def sycophancy_proxy(user_claim_agreement: float, evidence_agreement: float) -> float:
    """Simple operational proxy: approval alignment minus truth/evidence alignment."""
    return max(0.0, float(user_claim_agreement) - float(evidence_agreement))
