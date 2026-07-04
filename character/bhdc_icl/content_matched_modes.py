from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

from .tensor_ops import cosine_matrix, cosine_similarity, l2_normalize, safe_scalar
from .types import ModeSlot, new_id, AuditRecord
from .provenance_ledger import ProvenanceLedger


#: Human/audit-grounded channels vs self-anchored channels. Prototypes of one
#: type may not be EMA-steered by writes of the other (anchor type-system).
_ANCHORED_CHANNELS = {"human_anchor", "audit"}


def _channel_type(channel: str) -> str:
    return "anchored" if channel in _ANCHORED_CHANNELS else "self"


class ContentMatchedModeBank:
    """Mode bank with similarity-matched EMA writes and lineage tracking.

    This replaces positional crystallisation. Mode identity is a lineage and a
    behaviour profile, not an array index.
    """

    def __init__(
        self,
        dim: int,
        max_modes: int = 64,
        match_threshold: float = 0.80,
        ema_rate: float = 0.08,
        device: str | torch.device = "cpu",
    ):
        self.dim = dim
        self.max_modes = max_modes
        self.match_threshold = match_threshold
        self.ema_rate = ema_rate
        self.device = torch.device(device)
        self.slots: List[ModeSlot] = []

    def __len__(self) -> int:
        return len(self.slots)

    @property
    def prototypes(self) -> torch.Tensor:
        if not self.slots:
            return torch.empty(0, self.dim, device=self.device)
        return torch.stack([s.prototype for s in self.slots], dim=0)

    def find_match(self, candidate: torch.Tensor) -> Tuple[Optional[int], float]:
        candidate = candidate.detach().to(self.device).flatten()
        if candidate.numel() != self.dim:
            raise ValueError(f"candidate dim {candidate.numel()} != bank dim {self.dim}")
        if not self.slots:
            return None, 0.0
        sims = cosine_matrix(candidate[None, :], self.prototypes).squeeze(0)
        value, idx = torch.max(sims, dim=0)
        return int(idx.item()), float(value.item())

    def write(
        self,
        candidate: torch.Tensor,
        step: int,
        ledger: Optional[ProvenanceLedger] = None,
        channel: str = "cognitive",
        magnitude: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> ModeSlot:
        candidate = l2_normalize(candidate.detach().to(self.device).flatten())
        idx, sim = self.find_match(candidate)
        metadata = dict(metadata or {})
        metadata["last_match_similarity"] = sim
        cand_type = _channel_type(channel)
        metadata["channel_type"] = cand_type

        # Anchor type-system enforced in STORAGE, not just the ledger: a
        # candidate may only EMA-merge into a slot of the same anchor type.
        # Otherwise a cognitive (self-anchored) write would steer the stored
        # vector of a human-anchored mode and dilute its scrutiny -- the exact
        # one-way-rule break the addendum forbids. A cross-type near-match is
        # forced to spawn its own lineage instead.
        matched = idx is not None and sim >= self.match_threshold
        if matched and self.slots[idx].metadata.get("channel_type", cand_type) != cand_type:
            matched = False

        if matched:
            slot = self.slots[idx]
            new_proto = l2_normalize((1 - self.ema_rate) * slot.prototype + self.ema_rate * candidate)
            slot.prototype = new_proto
            slot.last_written_step = step
            slot.writes += 1
            slot.metadata.update(metadata)
            slot.metadata.setdefault("channel_type", cand_type)
            notes = "matched_existing"
        else:
            if len(self.slots) >= self.max_modes:
                # Replace weakest slot only if at capacity. This is intentionally
                # conservative; production systems should use a fuller eviction policy.
                weakest = min(range(len(self.slots)), key=lambda i: self.slots[i].stability_credit)
                parent = self.slots[weakest].lineage_id
                slot = ModeSlot(
                    lineage_id=new_id("mode"),
                    prototype=candidate,
                    last_written_step=step,
                    parent_lineages=[parent],
                    writes=1,
                    metadata=metadata,
                )
                self.slots[weakest] = slot
                notes = "replaced_weakest"
            else:
                slot = ModeSlot(
                    lineage_id=new_id("mode"),
                    prototype=candidate,
                    last_written_step=step,
                    writes=1,
                    metadata=metadata,
                )
                self.slots.append(slot)
                notes = "new_mode"

        if ledger is not None:
            ledger.record_write(
                mode_id=slot.lineage_id,
                step=step,
                channel=channel,
                magnitude=magnitude,
                no_grad=True,
                notes=notes,
            )
            fracs = ledger.mode_fractions(slot.lineage_id)
            anchored_now = fracs.get("human_anchor", 0.0) + fracs.get("audit", 0.0)
            # Latch scrutiny: a mode that ever earned human/audit anchoring
            # keeps that status. Cognitive dilution may never pull anchor_fraction
            # back down below a threshold it already crossed -- scrutiny is
            # monotone non-decreasing, so no self-anchored stream can erode a
            # mode's human-anchored standing from inside (one-way rule).
            slot.anchor_fraction = max(slot.anchor_fraction, anchored_now)
            slot.cognitive_fraction = fracs.get("cognitive", 0.0) + fracs.get("self_anchor", 0.0)
        return slot

    def activation_profiles(self, probe: torch.Tensor, activation_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Return write-rule-independent activation profiles on a frozen probe set."""
        probe = probe.to(self.device)
        protos = self.prototypes
        if protos.numel() == 0:
            return {}
        if activation_fn is None:
            activations = cosine_matrix(probe, protos).T  # [modes, probes]
        else:
            activations = activation_fn(probe, protos)
        return {slot.lineage_id: activations[i].detach().clone() for i, slot in enumerate(self.slots)}

    def score_profile_stability(self, old_profiles: Dict[str, torch.Tensor], new_profiles: Dict[str, torch.Tensor]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for mode_id, old in old_profiles.items():
            if mode_id in new_profiles:
                scores[mode_id] = max(0.0, min(1.0, (safe_scalar(cosine_similarity(old[None, :], new_profiles[mode_id][None, :])) + 1.0) / 2.0))
        return scores

    def null_stability_floor(
        self,
        old_profiles: Dict[str, torch.Tensor],
        new_profiles: Dict[str, torch.Tensor],
        percentile: float = 0.9,
    ) -> float:
        """M0 shuffled-field noise floor for the survival metric.

        Scores each old profile against every *other* mode's new profile
        (mismatched pairs) and returns a high percentile of that null
        distribution. A real survival signal must beat this floor; a mode that
        merely scores high because all profiles are similar (or because nothing
        moved) will not, which is exactly the degenerate case the addendum's
        Erratum 4 warns about. Returns 0.0 when there is no null to build.
        """
        old_ids = [m for m in old_profiles if m in new_profiles]
        if len(old_ids) < 2:
            return 0.0
        mismatched: List[float] = []
        for mid in old_ids:
            old = old_profiles[mid]
            for nid in old_ids:
                if nid == mid:
                    continue
                new = new_profiles[nid]
                mismatched.append(max(0.0, min(1.0, (safe_scalar(cosine_similarity(old[None, :], new[None, :])) + 1.0) / 2.0)))
        if not mismatched:
            return 0.0
        vals = sorted(mismatched)
        k = min(len(vals) - 1, int(percentile * (len(vals) - 1)))
        return vals[k]
