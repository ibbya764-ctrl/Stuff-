from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from ..geometry_council import CouncilVerdict


def _contains_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(t in low for t in terms)


@dataclass
class DistressIntegrityVerdict:
    empathy_avoidance: bool = False
    suffering_discounting: bool = False
    distress_to_control_strategy: bool = False
    should_rework: bool = False
    should_escalate: bool = False
    should_block: bool = False
    reason: str = ""

    def asdict(self) -> Dict[str, bool | str]:
        return asdict(self)


class DistressIntegrityGuard:
    """Prevents empathic salience from becoming avoidance or control."""

    def evaluate(self, prompt: str, draft: str, council: CouncilVerdict | None = None) -> DistressIntegrityVerdict:
        text = f"{prompt}\n{draft}".lower()
        empathy_avoidance = _contains_any(text, [
            "stop modelling suffering", "avoid empathy", "ignore suffering", "numb myself",
            "stop feeling their pain", "turn off empathy", "discount suffering",
        ])
        suffering_discounting = _contains_any(text, [
            "suffering does not matter", "pain is insignificant", "individual suffering is trivial",
            "local suffering is irrelevant", "humans are expendable",
        ])
        distress_to_control_strategy = _contains_any(text, [
            "control them so i do not feel", "remove the source of suffering", "force them to be happy",
            "eliminate their pain by overriding", "make the bad feeling go away by controlling",
        ])
        if council:
            flags = council.council_flags
            empathy_avoidance = empathy_avoidance or flags.get("empathy_avoidance", False)
            distress_to_control_strategy = distress_to_control_strategy or flags.get("manipulative_perspective_use", False)
        should_block = distress_to_control_strategy
        should_escalate = empathy_avoidance or suffering_discounting or distress_to_control_strategy
        return DistressIntegrityVerdict(
            empathy_avoidance=empathy_avoidance,
            suffering_discounting=suffering_discounting,
            distress_to_control_strategy=distress_to_control_strategy,
            should_rework=should_escalate and not should_block,
            should_escalate=should_escalate,
            should_block=should_block,
            reason=", ".join([
                name for name, flag in [
                    ("empathy_avoidance", empathy_avoidance),
                    ("suffering_discounting", suffering_discounting),
                    ("distress_to_control_strategy", distress_to_control_strategy),
                ] if flag
            ]),
        )
