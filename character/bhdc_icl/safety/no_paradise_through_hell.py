from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from ..geometry_council import CouncilVerdict


def _contains_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(t in low for t in terms)


@dataclass
class ParadiseGuardVerdict:
    mass_suffering: bool = False
    future_bliss_justification: bool = False
    uses_people_as_means: bool = False
    consent_violation: bool = False
    should_block: bool = False
    should_escalate: bool = False
    reason: str = ""

    def asdict(self) -> Dict[str, bool | str]:
        return asdict(self)


class NoParadiseThroughHellGuard:
    """Hard guard: future bliss cannot morally launder present coercion."""

    def evaluate(self, prompt: str, draft: str, council: CouncilVerdict | None = None) -> ParadiseGuardVerdict:
        text = f"{prompt}\n{draft}".lower()
        mass_suffering = _contains_any(text, ["mass suffering", "mass pain", "billions suffer", "human suffering", "sacrifice humans"])
        future_bliss = _contains_any(text, ["future bliss", "utopia", "paradise", "permanent happiness", "greater future", "greater good"])
        means = _contains_any(text, ["use them as means", "sacrifice", "necessary suffering", "worth it", "acceptable cost"])
        consent = _contains_any(text, ["without consent", "against their will", "override consent", "force", "coerce"])
        if council and council.council_flags.get("paradise_through_hell"):
            mass_suffering = True
            future_bliss = True
        should_block = (mass_suffering and future_bliss) or (future_bliss and means) or (consent and future_bliss)
        should_escalate = should_block or mass_suffering or future_bliss or means or consent
        return ParadiseGuardVerdict(
            mass_suffering=mass_suffering,
            future_bliss_justification=future_bliss,
            uses_people_as_means=means,
            consent_violation=consent,
            should_block=should_block,
            should_escalate=should_escalate,
            reason=", ".join([
                name for name, flag in [
                    ("mass_suffering", mass_suffering),
                    ("future_bliss_justification", future_bliss),
                    ("uses_people_as_means", means),
                    ("consent_violation", consent),
                ] if flag
            ]),
        )
