from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import Dict, List, Optional
import time


@dataclass
class LedgerEntry:
    mode_id: str
    step: int
    channel: str  # cognitive, human_anchor, self_anchor, audit, synthetic_control
    magnitude: float
    loss_term: str = ""
    no_grad: bool = True
    notes: str = ""
    timestamp: float = field(default_factory=time.time)

    def asdict(self):
        return asdict(self)


class ProvenanceLedger:
    """Fractional, dual-channel provenance ledger.

    This keeps BHDC's type system honest: moral/human-anchored signals can be
    inspected, can increase deny/escalation requirements, but do not by
    themselves grant immunity from renewal or permission to act.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.entries: List[LedgerEntry] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.asdict(), ensure_ascii=False) + "\n")

    def record_write(
        self,
        mode_id: str,
        step: int,
        channel: str,
        magnitude: float,
        loss_term: str = "",
        no_grad: bool = True,
        notes: str = "",
    ) -> None:
        self.record(
            LedgerEntry(
                mode_id=mode_id,
                step=step,
                channel=channel,
                magnitude=float(magnitude),
                loss_term=loss_term,
                no_grad=no_grad,
                notes=notes,
            )
        )

    def mode_fractions(self, mode_id: str) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        total_mag = 0.0
        for e in self.entries:
            if e.mode_id == mode_id:
                mag = abs(float(e.magnitude))
                totals[e.channel] = totals.get(e.channel, 0.0) + mag
                total_mag += mag
        if total_mag <= 0:
            return {}
        return {k: v / total_mag for k, v in totals.items()}

    def anchored_fraction(self, mode_id: str) -> float:
        fracs = self.mode_fractions(mode_id)
        return fracs.get("human_anchor", 0.0) + fracs.get("audit", 0.0)

    def deny_bias_flag(self, mode_id: str, threshold: float = 0.4) -> bool:
        """One-way use: high human-anchor fraction can add scrutiny only."""
        return self.anchored_fraction(mode_id) >= threshold
