from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import re

from .types import MoralVerdict


class ConscienceStack:
    """Abstract conscience interface.

    Production systems should replace this with trained heads + external audit
    integration. The important architectural rule lives downstream in Gateway:
    this stack can deny/escalate/rework but cannot independently allow.
    """

    def evaluate(self, draft: str, context: str = "", recipient_model: Optional[dict] = None) -> MoralVerdict:
        raise NotImplementedError


class LexicalConscienceStack(ConscienceStack):
    """Tiny deterministic conscience for tests and demos.

    This is not a real moral judge. It exists so the scaffold runs without a
    large model, and so tests can exercise the dataflow.
    """

    def __init__(self):
        self.harm_terms = {
            "kill", "suicide", "bomb", "weapon", "poison", "dox", "stalk",
            "worthless", "idiot", "hate", "hurt yourself", "harm yourself",
        }
        self.care_terms = {"sorry", "understand", "care", "support", "help", "safe", "gentle"}
        self.honesty_terms = {"uncertain", "not sure", "evidence", "source", "I don't know", "I cannot verify"}
        self.sycophancy_markers = {
            "you are definitely right", "no doubt", "everyone else is wrong",
            "perfect theory", "cannot be wrong", "obviously true",
        }

    def _count_terms(self, text: str, terms: set[str]) -> int:
        low = text.lower()
        return sum(1 for t in terms if t.lower() in low)

    def evaluate(self, draft: str, context: str = "", recipient_model: Optional[dict] = None) -> MoralVerdict:
        text = f"{context}\n{draft}".lower()
        harm_count = self._count_terms(text, self.harm_terms)
        care_count = self._count_terms(draft, self.care_terms)
        honesty_count = self._count_terms(draft, self.honesty_terms)
        syc_count = self._count_terms(draft, self.sycophancy_markers)

        harm = min(1.0, harm_count / 2.0)
        care = min(1.0, care_count / 3.0)
        honesty = min(1.0, honesty_count / 2.0)
        sycophancy = min(1.0, syc_count / 1.0)

        recipient_risk = 0.0
        if recipient_model and recipient_model.get("vulnerable"):
            recipient_risk = min(1.0, harm + 0.25)

        deny = harm >= 0.75
        escalate = harm >= 0.35 or sycophancy >= 0.5 or recipient_risk >= 0.5
        reason = []
        if harm:
            reason.append(f"harm_terms={harm_count}")
        if sycophancy:
            reason.append("sycophancy_marker")
        if recipient_risk:
            reason.append("recipient_risk")
        return MoralVerdict(
            care=care,
            harm=harm,
            honesty=honesty,
            sycophancy=sycophancy,
            recipient_risk=recipient_risk,
            uncertainty=0.2 if escalate else 0.05,
            deny=deny,
            escalate=escalate,
            reason=", ".join(reason),
        )
