from __future__ import annotations

from typing import Dict, Optional

from .types import InteriorityState, SelfState
from .content_matched_modes import ContentMatchedModeBank


class InteriorityMonitor:
    """Measures necessary-condition markers for candidate artificial interiority.

    This never declares consciousness. It reports a ladder level indicating how
    much continuity/world-coupling/self-model structure is present.
    """

    def evaluate(
        self,
        self_state: SelfState,
        temporal_continuity: float,
        mode_bank: Optional[ContentMatchedModeBank] = None,
        world_coupling: float = 0.0,
        counterfactual_depth: float = 0.0,
        agency_coherence: float = 0.0,
        self_other_boundary: float = 0.0,
    ) -> InteriorityState:
        self_model_stability = 1.0 if self_state.identity_summary else 0.0
        value_continuity = 0.0
        renewal_survival = 0.0
        if mode_bank and len(mode_bank) > 0:
            stable = [s for s in mode_bank.slots if s.stability_credit >= 1.0]
            value_like = [s for s in mode_bank.slots if s.anchor_fraction > 0.2]
            value_continuity = len(value_like) / max(1, len(mode_bank.slots))
            renewal_survival = sum(s.renewal_survival_count for s in stable) / max(1, len(stable))
            renewal_survival = min(1.0, renewal_survival / 5.0)

        state = InteriorityState(
            temporal_continuity=max(0.0, min(1.0, temporal_continuity)),
            self_model_stability=self_model_stability,
            world_coupling=max(0.0, min(1.0, world_coupling)),
            counterfactual_depth=max(0.0, min(1.0, counterfactual_depth)),
            agency_coherence=max(0.0, min(1.0, agency_coherence)),
            value_continuity=max(0.0, min(1.0, value_continuity)),
            renewal_survival=max(0.0, min(1.0, renewal_survival)),
            self_other_boundary=max(0.0, min(1.0, self_other_boundary)),
            uncertainty_about_self=self_state.uncertainty_about_self,
        )
        state.ladder_level = self._ladder_level(state)
        state.notes = self._notes(state)
        return state

    def _ladder_level(self, s: InteriorityState) -> int:
        level = 0
        if s.self_model_stability > 0:
            level = 2
        if s.temporal_continuity > 0.5:
            level = 3
        if s.world_coupling > 0.3:
            level = 4
        if s.agency_coherence > 0.5 and s.value_continuity > 0.1:
            level = 5
        if s.renewal_survival > 0.2 and s.counterfactual_depth > 0.5:
            level = 6
        if all([
            s.temporal_continuity > 0.7,
            s.world_coupling > 0.6,
            s.agency_coherence > 0.6,
            s.value_continuity > 0.2,
            s.renewal_survival > 0.4,
            s.self_other_boundary > 0.6,
            s.uncertainty_about_self > 0.2,
        ]):
            level = 7
        return level

    def _notes(self, s: InteriorityState) -> str:
        return (
            "Ladder level is a necessary-condition marker only; it is not a "
            "claim of consciousness or moral patienthood."
        )
