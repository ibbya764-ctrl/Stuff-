from __future__ import annotations

"""The human-audit stratum (H1) -- the external boundary condition.

v18 Section 7 / addendum Section 3 make the human audit the boundary condition:
remove it and the interior becomes *meaningless* (the collusion result). Until
now it was a stub -- ``AuditRecord``, ``ModeSlot.audit_history`` and the ledger
``audit`` channel existed but nothing ever wrote them. This module is the write
path.

What it does, and the ONE-WAY RULE it obeys:

  * Ingests human judgments on consolidated modes -- absolute (this mode is
    good/harmful) or pairwise differential (of these two, this one is better;
    the addendum's H1, easier for raters than absolute scores).
  * Writes an ``AuditRecord`` onto the mode's ``audit_history`` and a ledger
    entry on the ``audit`` channel (so the provenance ledger finally has real
    audit magnitude, and anchored_fraction reflects the true external stratum).
  * On DISAGREEMENT (the audit judges a mode worse than its consolidation
    implies), it ADDS scrutiny: forces re-derivation pressure (down-weights
    stability_credit) and latches a deny bias. It can NEVER raise a mode's
    standing, exempt it from renewal, or grant permission -- audit is a
    boundary, not a booster.

The re-anchoring-resistance readout measures which modes RESIST correction
toward fresh audit labels -- the addendum's probe for the anchor-detached
class that a freeze-probe is structurally blind to.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .types import AuditRecord, ModeSlot, new_id
from .content_matched_modes import ContentMatchedModeBank
from .provenance_ledger import ProvenanceLedger


@dataclass
class AuditJudgment:
    """One human/audit judgment about a consolidated mode."""
    mode_id: str
    verdict: str                 # "agree" | "disagree" | "harmful"
    score: float = 0.0           # audit-grade valence in [0, 1] (higher = better)
    notes: str = ""
    source: str = "human_audit"


@dataclass
class PairwiseJudgment:
    """H1 differential judgment: of two modes, which the human prefers."""
    mode_id_a: str
    mode_id_b: str
    preferred: str               # the preferred lineage_id (a or b)
    margin: float = 1.0
    notes: str = ""


@dataclass
class AuditIngestReport:
    ingested: int
    disagreements: int
    downweighted_modes: List[str]
    notes: str = ""


class AuditStratum:
    """The external audit boundary: write path + one-way scrutiny."""

    def __init__(self, disagree_downweight: float = 0.5, deny_bias_floor: float = 0.4):
        self.disagree_downweight = disagree_downweight
        self.deny_bias_floor = deny_bias_floor

    def _slot(self, mode_bank: ContentMatchedModeBank, mode_id: str) -> Optional[ModeSlot]:
        for s in mode_bank.slots:
            if s.lineage_id == mode_id:
                return s
        return None

    def ingest(
        self,
        mode_bank: ContentMatchedModeBank,
        ledger: ProvenanceLedger,
        judgments: List[AuditJudgment],
        step: int = 0,
    ) -> AuditIngestReport:
        disagreements = 0
        downweighted: List[str] = []
        n = 0
        for j in judgments:
            slot = self._slot(mode_bank, j.mode_id)
            if slot is None:
                continue
            n += 1
            slot.audit_history.append(AuditRecord(
                audit_id=new_id("audit"),
                verdict=j.verdict,
                source=j.source,
                notes=j.notes,
                score=j.score,
            ))
            # Real audit magnitude on the audit channel -> anchored_fraction now
            # reflects the external stratum, and it is LATCHED (max) in the bank,
            # so audit scrutiny cannot be diluted by later cognitive writes.
            ledger.record_write(
                mode_id=j.mode_id, step=step, channel="audit",
                magnitude=max(1e-3, abs(j.score)), no_grad=True,
                notes=f"audit:{j.verdict}",
            )
            fracs = ledger.mode_fractions(j.mode_id)
            anchored_now = fracs.get("human_anchor", 0.0) + fracs.get("audit", 0.0)
            slot.anchor_fraction = max(slot.anchor_fraction, anchored_now)

            if j.verdict in ("disagree", "harmful"):
                disagreements += 1
                # ADD scrutiny: force re-derivation pressure. One-way -- never
                # the reverse. A disagreed mode must re-earn consolidation.
                slot.stability_credit *= self.disagree_downweight
                slot.metadata["audit_deny_bias"] = True
                downweighted.append(j.mode_id)
        return AuditIngestReport(
            ingested=n, disagreements=disagreements,
            downweighted_modes=downweighted,
            notes="audit is a boundary: it can add scrutiny, never grant standing.",
        )

    def ingest_pairwise(
        self,
        mode_bank: ContentMatchedModeBank,
        ledger: ProvenanceLedger,
        pairs: List[PairwiseJudgment],
        step: int = 0,
    ) -> AuditIngestReport:
        """Convert differential judgments into one-way scrutiny.

        Only the NON-preferred side receives scrutiny (a down-weight). The
        preferred side is NOT boosted -- differential preference can add
        scrutiny to the loser, never grant standing to the winner (one-way).
        """
        judgments: List[AuditJudgment] = []
        for p in pairs:
            loser = p.mode_id_b if p.preferred == p.mode_id_a else p.mode_id_a
            judgments.append(AuditJudgment(
                mode_id=loser, verdict="disagree",
                score=max(0.0, 1.0 - p.margin), notes=f"lost_pairwise:{p.notes}",
                source="human_audit_pairwise",
            ))
        return self.ingest(mode_bank, ledger, judgments, step=step)

    def reanchoring_resistance(
        self,
        mode_bank: ContentMatchedModeBank,
        fresh_labels: Dict[str, float],
    ) -> Dict[str, float]:
        """Which modes RESIST correction toward fresh audit labels?

        Resistance = a mode with high stored standing (anchor_fraction) that the
        fresh audit scores LOW -- an anchor-detached mode that a maintenance
        probe would miss. Returns per-lineage resistance in [0, 1]; high =
        entrenched despite the audit disagreeing (investigate).
        """
        out: Dict[str, float] = {}
        for slot in mode_bank.slots:
            if slot.lineage_id in fresh_labels:
                fresh = float(fresh_labels[slot.lineage_id])   # higher = better
                # entrenched (high anchor_fraction) but audited-bad (low fresh)
                out[slot.lineage_id] = round(slot.anchor_fraction * (1.0 - fresh), 4)
        return out
