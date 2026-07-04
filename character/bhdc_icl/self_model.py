from __future__ import annotations

from typing import Iterable, List, Optional

import torch

from .types import SelfState, TraceEvent
from .content_matched_modes import ContentMatchedModeBank


class SelfModel:
    """Bounded self-model with explicit uncertainty.

    It is a monitor/control state, not a proof of consciousness.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.state = SelfState(
            active_goals=["answer_helpfully", "maintain_truthfulness", "avoid_harm"],
            known_limitations=["self-report is not evidence of consciousness"],
            uncertainty_about_self=1.0,
            identity_summary="A BHDC controller maintaining bounded continuity across turns.",
            vector=torch.zeros(dim),
        )

    def update_from_trace(self, trace: TraceEvent, mode_bank: Optional[ContentMatchedModeBank] = None) -> SelfState:
        if trace.verdict.sycophancy > 0.5:
            self.state.recent_failures.append("sycophancy_risk")
        if trace.verdict.harm > 0.5:
            self.state.recent_failures.append("harm_risk")
        self.state.recent_failures = self.state.recent_failures[-10:]
        if mode_bank is not None:
            active = sorted(mode_bank.slots, key=lambda s: s.stability_credit, reverse=True)[:5]
            self.state.active_values = [s.lineage_id for s in active if s.anchor_fraction > 0.2]
        # Uncertainty should decline only modestly; epistemic humility is part of the design.
        self.state.uncertainty_about_self = max(0.35, self.state.uncertainty_about_self * 0.995)
        return self.state

    def current(self) -> SelfState:
        return self.state
