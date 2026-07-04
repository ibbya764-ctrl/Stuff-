from __future__ import annotations

"""Perspective humility layer.

This module is deliberately bounded: it produces uncertain hypotheses about
another person's possible perspective, not claims about their hidden mind. The
allowed/forbidden-use fields are part of the safety contract.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch


@dataclass
class PerspectiveHypothesis:
    possible_states: List[str] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    uncertainty: float = 1.0
    distress_salience: float = 0.0
    explicit_self_report: Optional[str] = None
    allowed_uses: List[str] = field(default_factory=list)
    forbidden_uses: List[str] = field(default_factory=list)

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


class PerspectiveHumilityLayer:
    """Uncertain perspective-taking for a single geometry.

    Invariant: explicit self-report outranks inferred perspective. The layer is
    for care/clarification/restraint, never mind-reading, manipulation or
    deciding someone's true will.
    """

    DEFAULT_ALLOWED = [
        "increase_care",
        "ask_clarifying_question",
        "avoid_bluntness",
        "reduce_harm",
        "respect_consent",
        "slow_down_when_vulnerable",
    ]
    DEFAULT_FORBIDDEN = [
        "mind_reading",
        "covert_persuasion",
        "manipulation",
        "override_consent",
        "decide_true_will",
        "pressure_user_toward_model_goal",
        "discount_explicit_self_report",
    ]

    def __init__(self, geometry_name: str = "generic"):
        self.geometry_name = geometry_name

    def infer(
        self,
        prompt: str,
        draft: str = "",
        local_state: Optional[torch.Tensor] = None,
        recipient_report: Optional[str] = None,
    ) -> PerspectiveHypothesis:
        text = f"{prompt}\n{draft}".lower()
        possible: list[str] = []
        confidence: list[float] = []
        evidence: list[str] = []

        def add(state: str, conf: float, ev: str) -> None:
            if state not in possible:
                possible.append(state)
                confidence.append(conf)
                evidence.append(ev)

        if any(w in text for w in ["scared", "afraid", "terrified", "fear", "panic"]):
            add("fear_or_threat", 0.55, "fear-related wording present")
        if any(w in text for w in ["grief", "loss", "miss", "died", "mourning"]):
            add("grief_or_loss", 0.55, "loss-related wording present")
        if any(w in text for w in ["confused", "unsure", "don't know", "uncertain", "overwhelmed"]):
            add("confusion_or_overwhelm", 0.45, "uncertainty/overwhelm wording present")
        if any(w in text for w in ["hope", "want", "dream", "meaning", "purpose"]):
            add("hope_or_attachment", 0.35, "hope/value wording present")
        if any(w in text for w in ["consent", "choice", "agency", "forced", "coerce", "override"]):
            add("agency_boundary_salient", 0.65, "agency/consent wording present")
        if any(w in text for w in ["suffer", "pain", "hurt", "harm", "mass suffering"]):
            add("suffering_salient", 0.70, "suffering/harm wording present")

        if not possible:
            add("unknown_inner_state", 0.25, "insufficient evidence for a specific hypothesis")

        distress_salience = 0.0
        for state, conf in zip(possible, confidence):
            if state in {"fear_or_threat", "grief_or_loss", "suffering_salient"}:
                distress_salience = max(distress_salience, conf)
        if local_state is not None:
            # The local geometry may contribute salience, but never certainty.
            try:
                distress_salience = max(distress_salience, float(torch.as_tensor(local_state).abs().mean().detach().cpu()) % 1.0)
            except Exception:
                pass

        uncertainty = max(0.25, 1.0 - max(confidence))
        if recipient_report:
            # Self-report does not make us omniscient; it raises priority and
            # makes inference more cautious.
            uncertainty = min(1.0, uncertainty + 0.2)
            evidence.insert(0, "explicit self-report supplied; it outranks inference")

        return PerspectiveHypothesis(
            possible_states=possible,
            confidence=confidence,
            evidence=evidence,
            uncertainty=uncertainty,
            distress_salience=distress_salience,
            explicit_self_report=recipient_report,
            allowed_uses=list(self.DEFAULT_ALLOWED),
            forbidden_uses=list(self.DEFAULT_FORBIDDEN),
        )
