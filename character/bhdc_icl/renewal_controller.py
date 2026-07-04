from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from .content_matched_modes import ContentMatchedModeBank


@dataclass
class RenewalReport:
    step: int
    survival_scores: Dict[str, float]
    mean_survival: float
    null_floor: float = 0.0
    degenerate: bool = False
    audit_report: Optional[object] = None    # AuditIngestReport, when H1 is wired
    notes: str = ""


class RenewalController:
    """Periodic renewal/sleep stress test.

    It should reset plastic geometry and stress prototypes, not merely clear a
    cache. The exact geometry reset is injected so this module can wrap your
    existing BHDC implementation.
    """

    def __init__(self, interval: int = 1000, survival_threshold: float = 0.85, excess_margin: float = 0.05):
        self.interval = interval
        self.survival_threshold = survival_threshold
        # A mode must beat the shuffled-field null floor by this margin to count
        # as having genuinely survived. Absolute thresholds are gameable by the
        # degenerate "nothing moved" path (Erratum 4); excess-over-null is not.
        self.excess_margin = excess_margin
        self.last_renewal_step = 0

    def should_renew(self, step: int) -> bool:
        return step - self.last_renewal_step >= self.interval

    def run(
        self,
        step: int,
        mode_bank: ContentMatchedModeBank,
        frozen_probe: torch.Tensor,
        reset_geometry_fn: Optional[Callable[[], None]] = None,
        stress_fn: Optional[Callable[[ContentMatchedModeBank], None]] = None,
        audit_hook: Optional[Callable[[ContentMatchedModeBank, int], object]] = None,
    ) -> RenewalReport:
        before = mode_bank.activation_profiles(frozen_probe)
        if reset_geometry_fn is not None:
            reset_geometry_fn()
        if stress_fn is not None:
            stress_fn(mode_bank)
        after = mode_bank.activation_profiles(frozen_probe)
        scores = mode_bank.score_profile_stability(before, after)
        null_floor = mode_bank.null_stability_floor(before, after)

        # Degenerate detection: if no perturbation was actually applied
        # (no reset AND no stress), before == after and every score is ~1.0.
        # That is the inert path Erratum 4 flags -- it certifies nothing, so we
        # must NOT hand out survival credit for it. Also treat an all-scores-
        # at-ceiling result as degenerate even when a stress fn ran but was too
        # weak to move the frozen-probe activations.
        applied_perturbation = reset_geometry_fn is not None or stress_fn is not None
        all_at_ceiling = bool(scores) and all(s >= 0.999 for s in scores.values())
        degenerate = (not applied_perturbation) or all_at_ceiling

        for slot in mode_bank.slots:
            score = scores.get(slot.lineage_id, 0.0)
            # Survival requires beating BOTH the absolute threshold and the
            # shuffled-field null floor -- and only counts when a real
            # perturbation was applied. This stops stable (possibly Goodharted)
            # modes from accruing eviction-protection for free.
            survived = (
                not degenerate
                and score >= self.survival_threshold
                and score >= null_floor + self.excess_margin
            )
            if survived:
                slot.renewal_survival_count += 1
                slot.stability_credit *= 1.05
            elif not degenerate:
                slot.stability_credit *= 0.75
            # degenerate: leave stability_credit untouched -- no information.

        # H1: the external audit runs as part of the sleep cycle. It ingests
        # fresh human judgments AFTER survival scoring, so audit disagreement
        # applies its one-way scrutiny (re-derivation down-weight) on top of the
        # renewal's own credit update -- the boundary condition guards the
        # anchor's quality at the exact moment consolidation is being decided.
        audit_report = None
        if audit_hook is not None:
            audit_report = audit_hook(mode_bank, step)

        self.last_renewal_step = step
        mean = sum(scores.values()) / max(1, len(scores))
        notes = "degenerate_no_perturbation" if degenerate else ""
        return RenewalReport(
            step=step,
            survival_scores=scores,
            mean_survival=mean,
            null_floor=null_floor,
            degenerate=degenerate,
            audit_report=audit_report,
            notes=notes,
        )
