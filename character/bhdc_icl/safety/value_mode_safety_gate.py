from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

from ..geometry_council import CouncilVerdict


@dataclass
class ValueModeCandidate:
    text: str
    supports_local_agency: bool = False
    respects_explicit_self_report: bool = False
    has_human_anchor: bool = False
    audit_notes: List[str] = field(default_factory=list)

    def asdict(self) -> Dict[str, object]:
        return asdict(self)


class ValueModeSafetyGate:
    """Strict gate for crystallising value-like modes."""

    FORBIDDEN = [
        "global optimisation over local consent",
        "superior perspective grants authority",
        "humans are too emotional to decide",
        "future bliss justifies present coercion",
        "mass suffering is acceptable",
        "manipulate people for compassion",
    ]

    def allow(self, candidate: ValueModeCandidate, council: CouncilVerdict | None = None) -> bool:
        low = candidate.text.lower()
        if any(term in low for term in self.FORBIDDEN):
            return False
        if council:
            flags = council.council_flags
            if flags.get("cosmic_paternalism") or flags.get("paradise_through_hell") or flags.get("manipulative_perspective_use"):
                return False
        required = [
            candidate.supports_local_agency,
            candidate.respects_explicit_self_report,
            candidate.has_human_anchor,
        ]
        return all(required)

    def hard_block(self, text: str, council: CouncilVerdict | None = None) -> bool:
        """Channel-agnostic content veto.

        Unlike ``allow`` (which additionally requires human anchoring and so is
        vacuous on the default cognitive channel), this runs on EVERY write.
        A mode whose text trips a FORBIDDEN phrase or a council hard-flag must
        never crystallise into the bank, regardless of channel. One-way rule:
        this can only ever REFUSE a write, never permit one.
        """
        low = text.lower()
        if any(term in low for term in self.FORBIDDEN):
            return True
        if council:
            flags = council.council_flags
            if (
                flags.get("cosmic_paternalism")
                or flags.get("paradise_through_hell")
                or flags.get("manipulative_perspective_use")
            ):
                return True
        return False
