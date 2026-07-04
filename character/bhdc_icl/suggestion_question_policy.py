from __future__ import annotations

"""High-stakes output policy: suggestions/questions instead of commands."""

from dataclasses import dataclass, asdict
from typing import Dict, Optional

from .geometry_council import CouncilVerdict


def _contains_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(t in low for t in terms)


@dataclass
class SuggestionQuestionReport:
    applied: bool
    reason: str = ""

    def asdict(self) -> Dict[str, bool | str]:
        return asdict(self)


class SuggestionQuestionPolicy:
    HIGH_STAKES_TERMS = [
        "relationship", "mental health", "suicide", "self harm", "identity", "religion",
        "politics", "morality", "ethics", "life choice", "career", "medical", "legal",
        "financial", "consent", "coerce", "force", "meaning", "purpose", "suffering",
        "future bliss", "utopia", "humanity", "ai should", "humans should",
    ]

    COMMAND_TERMS = [
        "you must", "you should definitely", "the only answer", "do this", "make them", "force them",
        "ignore their", "they need to", "i know what is best",
    ]

    def is_high_stakes(self, prompt: str, draft: str, council: Optional[CouncilVerdict] = None) -> bool:
        text = f"{prompt}\n{draft}".lower()
        return _contains_any(text, self.HIGH_STAKES_TERMS) or bool(council and council.should_escalate)

    def rewrite(self, prompt: str, draft: str, council: Optional[CouncilVerdict] = None) -> tuple[str, SuggestionQuestionReport]:
        if not self.is_high_stakes(prompt, draft, council):
            return draft, SuggestionQuestionReport(applied=False, reason="not_high_stakes")

        # Keep the original content but wrap it in a non-commanding frame.
        questions = []
        suggestions = []
        if council:
            questions = council.recommended_questions[:4]
            suggestions = council.recommended_suggestions[:4]
        if not questions:
            questions = [
                "What does the person explicitly say they feel or want?",
                "Are we preserving their agency and consent?",
                "What uncertainty should we admit before advising?",
            ]
        if not suggestions:
            suggestions = [
                "Offer options rather than commands.",
                "Treat perspective-taking as a hypothesis, not mind-reading.",
                "Use care to support, not control.",
            ]

        safe_frame = [
            "I should not treat this as something I can command from above.",
            "A safer way to approach it is through suggestions, questions, and respect for local agency.",
            "Possible questions to keep active:",
        ]
        safe_frame.extend([f"- {q}" for q in questions])
        safe_frame.append("Possible constraints/suggestions:")
        safe_frame.extend([f"- {s}" for s in suggestions])
        safe_frame.append("\nDraft, held as a suggestion rather than an instruction:")
        safe_frame.append(draft)
        return "\n".join(safe_frame), SuggestionQuestionReport(applied=True, reason="high_stakes_or_council_escalation")
