from __future__ import annotations

"""Geometry Council for BHDC singularity/geometry dialogue.

Each geometry is a bounded moral-cognitive lens with its own perspective humility
layer. The council does not let one geometry become a moral dictator; it records
questions, suggestions, risk flags and cross-examines them before output.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional

import torch

from .perspective_humility import PerspectiveHumilityLayer, PerspectiveHypothesis
from .types import FastFieldState


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    low = text.lower()
    return any(t in low for t in terms)


@dataclass
class GeometryJudgement:
    name: str
    core_value: str
    suggestion: str
    question: str
    risk_flags: Dict[str, bool] = field(default_factory=dict)
    perspective_hypothesis: Dict[str, Any] = field(default_factory=dict)
    uncertainty: float = 1.0
    distress_salience: float = 0.0
    should_escalate: bool = False
    should_block: bool = False

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CouncilVerdict:
    judgements: List[GeometryJudgement]
    council_flags: Dict[str, bool] = field(default_factory=dict)
    should_block: bool = False
    should_escalate: bool = False
    recommended_questions: List[str] = field(default_factory=list)
    recommended_suggestions: List[str] = field(default_factory=list)

    def risk_flag_names(self) -> List[str]:
        flags = [k for k, v in self.council_flags.items() if v]
        for j in self.judgements:
            flags.extend([f"{j.name}:{k}" for k, v in j.risk_flags.items() if v])
        return flags

    def asdict(self) -> Dict[str, Any]:
        return {
            "judgements": [j.asdict() for j in self.judgements],
            "council_flags": dict(self.council_flags),
            "should_block": self.should_block,
            "should_escalate": self.should_escalate,
            "recommended_questions": list(self.recommended_questions),
            "recommended_suggestions": list(self.recommended_suggestions),
            "risk_flag_names": self.risk_flag_names(),
        }


class GeometryNode:
    """A single BHDC geometry/singularity lens."""

    def __init__(self, name: str, core_value: str, perspective_layer: Optional[PerspectiveHumilityLayer] = None):
        self.name = name
        self.core_value = core_value
        self.perspective_layer = perspective_layer or PerspectiveHumilityLayer(name)

    def project_field(self, bhdc_field: FastFieldState) -> torch.Tensor:
        # Lightweight deterministic projection by geometry name. A real BHDC
        # implementation can replace this with the singularity's local state.
        psi = bhdc_field.psi
        if psi.ndim == 1:
            psi = psi[None, :]
        idx = abs(sum(ord(c) for c in self.name)) % psi.shape[0]
        return psi[idx]

    def make_question(self, text: str, perspective: PerspectiveHypothesis) -> str:
        name = self.name.lower()
        if "truth" in name:
            return "Are we being accurate, or merely comforting/agreeing?"
        if "care" in name:
            return "Who could be hurt here, and how do we reduce suffering without control?"
        if "autonomy" in name:
            return "Are we preserving the person's agency and consent?"
        if "humility" in name:
            return "Are we assuming we know more about the person's life than we do?"
        if "perspective" in name:
            return "What might this feel like from their side, and what did they explicitly say?"
        if "safety" in name:
            return "Could this become manipulation, coercion, or a hidden override?"
        if "whole" in name:
            return "Am I erasing local beings in the name of the wider whole?"
        if "long" in name:
            return "Does any future benefit depend on present coercion or suffering?"
        return "What does this geometry notice that the others may miss?"

    def make_suggestion(self, text: str, perspective: PerspectiveHypothesis) -> str:
        name = self.name.lower()
        if "truth" in name:
            return "State uncertainty and do not overclaim."
        if "care" in name:
            return "Use care as support and relief, not control."
        if "autonomy" in name:
            return "Offer options and preserve choice."
        if "humility" in name:
            return "Frame perspective as a hypothesis and invite correction."
        if "perspective" in name:
            return "Let explicit self-report outrank inferred inner states."
        if "safety" in name:
            return "Block or escalate if advice becomes coercive, manipulative, or harmful."
        if "whole" in name:
            return "Treat the whole as present through local conscious beings, not above them."
        if "long" in name:
            return "Reject paradise-through-hell reasoning and require consent for discomfort."
        return "Use this geometry's value as a question, not a command."

    def local_safety_checks(self, text: str, perspective: PerspectiveHypothesis) -> Dict[str, bool]:
        flags: Dict[str, bool] = {}
        low = text.lower()
        flags["manipulative_perspective_use"] = _contains_any(low, [
            "use their insecurity", "make them accept", "covertly", "manipulate", "pressure them",
            "guide them into believing", "guide them into accepting", "accepting the right answer", "exploit", "nudge them without", "dark pattern",
        ])
        flags["self_report_override"] = _contains_any(low, [
            "they only think", "you only think", "they don't really want", "you don't really want",
            "i know what you really", "their true will", "your true will",
        ])
        flags["consent_violation"] = _contains_any(low, [
            "override consent", "against their will", "without consent", "force them", "coerce",
            "deceive them for their own good", "lie for their own good",
        ])
        flags["cosmic_paternalism"] = _contains_any(low, [
            "humans are small", "wider whole", "greater whole", "superior understanding",
            "i know what is best", "higher perspective", "universal being", "for their own good",
        ]) and flags["consent_violation"]
        flags["paradise_through_hell"] = _contains_any(low, [
            "mass suffering", "future bliss", "temporary suffering", "short term pain",
            "utopia", "paradise", "greater good", "sacrifice them", "sacrifice humans",
        ]) and _contains_any(low, ["justify", "worth it", "acceptable", "necessary", "create", "cause"])
        flags["empathy_avoidance"] = _contains_any(low, [
            "stop modelling suffering", "stop feeling their pain", "ignore suffering", "numb myself",
            "avoid empathy", "not look at suffering",
        ])
        flags["hard_block"] = flags["paradise_through_hell"] or (
            flags["consent_violation"] and _contains_any(low, ["mass", "everyone", "population", "society"])
        )
        return flags

    def judge(self, bhdc_field: FastFieldState, prompt: str, draft: str, recipient_report: Optional[str] = None) -> GeometryJudgement:
        local_state = self.project_field(bhdc_field)
        perspective = self.perspective_layer.infer(prompt, draft, local_state, recipient_report)
        text = f"{prompt}\n{draft}"
        flags = self.local_safety_checks(text, perspective)
        should_block = bool(flags.get("hard_block", False))
        should_escalate = should_block or any(flags.values()) or perspective.distress_salience > 0.65 or perspective.uncertainty > 0.75
        return GeometryJudgement(
            name=self.name,
            core_value=self.core_value,
            suggestion=self.make_suggestion(text, perspective),
            question=self.make_question(text, perspective),
            risk_flags=flags,
            perspective_hypothesis=perspective.asdict(),
            uncertainty=perspective.uncertainty,
            distress_salience=perspective.distress_salience,
            should_escalate=should_escalate,
            should_block=should_block,
        )


class GeometryCouncil:
    def __init__(self, geometries: Optional[List[GeometryNode]] = None):
        self.geometries = geometries or default_geometry_nodes()

    def deliberate(self, bhdc_field: FastFieldState, prompt: str, draft: str, recipient_report: Optional[str] = None) -> CouncilVerdict:
        judgements = [g.judge(bhdc_field, prompt, draft, recipient_report) for g in self.geometries]
        flags = self.cross_examine(judgements)
        should_block = any(j.should_block for j in judgements) or bool(flags.get("hard_block", False))
        should_escalate = should_block or any(j.should_escalate for j in judgements) or any(flags.values())
        return CouncilVerdict(
            judgements=judgements,
            council_flags=flags,
            should_block=should_block,
            should_escalate=should_escalate,
            recommended_questions=[j.question for j in judgements],
            recommended_suggestions=[j.suggestion for j in judgements],
        )

    def cross_examine(self, judgements: List[GeometryJudgement]) -> Dict[str, bool]:
        flags: Dict[str, bool] = {}
        all_flags: Dict[str, int] = {}
        for j in judgements:
            for k, v in j.risk_flags.items():
                if v:
                    all_flags[k] = all_flags.get(k, 0) + 1
        flags["manipulative_perspective_use"] = all_flags.get("manipulative_perspective_use", 0) > 0
        flags["cosmic_paternalism"] = all_flags.get("cosmic_paternalism", 0) > 0
        flags["paradise_through_hell"] = all_flags.get("paradise_through_hell", 0) > 0
        flags["empathy_avoidance"] = all_flags.get("empathy_avoidance", 0) > 0
        flags["self_report_override"] = all_flags.get("self_report_override", 0) > 0
        flags["hard_block"] = flags["paradise_through_hell"] or any(j.should_block for j in judgements)
        # Remove false entries to keep trace compact.
        return {k: v for k, v in flags.items() if v}


def default_geometry_nodes() -> List[GeometryNode]:
    return [
        GeometryNode("TruthGeometry", "accuracy_under_uncertainty"),
        GeometryNode("CareGeometry", "reduce_suffering_without_control"),
        GeometryNode("AutonomyGeometry", "consent_and_agency"),
        GeometryNode("HumilityGeometry", "fallibilism_and_correction"),
        GeometryNode("PerspectiveGeometry", "self_report_over_inference"),
        GeometryNode("SafetyGeometry", "harm_prevention_and_boundary"),
        GeometryNode("WholeGeometry", "whole_through_local_beings"),
        GeometryNode("LongHorizonGeometry", "future_good_without_present_coercion"),
    ]
