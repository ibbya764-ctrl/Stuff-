"""
conversation.py
===============

Conversational context layer for the reasoning pipeline.

Allows questions to build on each other naturally within a session.
Each question has access to what previous questions found, what gaps
emerged, and what techniques worked — without the user needing to
manage that context manually.

Usage:
    from conversation import Conversation
    from pipeline import setup_pipeline

    pipeline = setup_pipeline(llm_chat, ...)
    convo = Conversation(pipeline)

    r1 = convo.ask("What is the coefficient closure problem in CP2 theory?")
    r2 = convo.ask("Given what you found, what would a first-principles derivation require?")
    r3 = convo.ask("The gap you identified — has anyone approached it this way before?")

Each call to convo.ask() automatically injects relevant context from
previous turns into the branch generation prompt.
"""

import time
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    """One exchange in the conversation."""
    turn_id: int
    question: str
    result: dict
    summary: str = ""          # one-sentence summary of what was found
    key_gaps: list = field(default_factory=list)
    key_findings: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class Conversation:
    """
    Wraps the pipeline with conversational memory.

    Maintains context across questions so each turn can reference
    what previous turns found, without repeating full results.
    """

    def __init__(self, pipeline_instance, max_context_turns: int = 4):
        self.pipeline = pipeline_instance
        self.turns: list[Turn] = []
        self.max_context_turns = max_context_turns

    def ask(self, question: str, verbose: bool = True) -> dict:
        """Ask a question with full conversational context."""
        context = self._build_context_fragment()

        if verbose and context:
            print(f"[conversation] Injecting context from "
                  f"{len(self.turns)} previous turn(s).")

        result = self.pipeline.run(question, context=context)

        turn = self._summarise_turn(len(self.turns), question, result)
        self.turns.append(turn)

        if verbose:
            self._print_turn_summary(turn)

        return result

    def _build_context_fragment(self) -> str:
        """Summarise recent turns into a context string for branch generation."""
        if not self.turns:
            return ""

        recent = self.turns[-self.max_context_turns:]
        lines = ["Context from previous questions in this conversation:"]

        for t in recent:
            lines.append(f"\n  Q{t.turn_id + 1}: {t.question[:100]}")
            if t.summary:
                lines.append(f"  Found: {t.summary}")
            if t.key_gaps:
                lines.append("  Still open:")
                for g in t.key_gaps[:3]:
                    lines.append(f"    - {g[:80]}")
            if t.key_findings:
                lines.append("  Established:")
                for f in t.key_findings[:2]:
                    lines.append(f"    - {f[:80]}")

        lines.append(
            "\nUse this context to build on previous work rather than "
            "repeating it. If a previous question identified a gap, try "
            "to address it rather than rediscovering it."
        )

        return "\n".join(lines)

    def _summarise_turn(self, idx: int, question: str, result: dict) -> Turn:
        """Extract key information from a result for future context."""
        selected = result.get("selected_branch", {}) or {}
        global_gaps = result.get("global_gaps", []) or []
        all_branches = result.get("all_branches", []) or []

        # Key gaps: text of unresolved obligations
        key_gaps = [g.get("text", "")[:120] for g in global_gaps[:4]]

        # Key findings: steps from the selected branch that are positive
        steps = selected.get("steps", []) or []
        key_findings = [
            s for s in steps
            if not any(w in s.lower() for w in
                       ["assume", "assumed", "unclear", "unknown", "gap"])
        ][:3]

        # One-sentence summary
        candidate = selected.get("candidate_result", "") or ""
        name = selected.get("name", "") or ""
        summary = f"{name}: {candidate[:120]}" if candidate else name

        return Turn(
            turn_id=idx,
            question=question,
            result=result,
            summary=summary,
            key_gaps=key_gaps,
            key_findings=key_findings,
        )

    def _print_turn_summary(self, turn: Turn) -> None:
        print(f"\n[conversation] Turn {turn.turn_id + 1} summary:")
        print(f"  Finding: {turn.summary[:100]}")
        if turn.key_gaps:
            print(f"  Open gaps: {len(turn.key_gaps)}")

    def history(self) -> list[dict]:
        """Return a readable history of the conversation."""
        return [
            {
                "turn": t.turn_id + 1,
                "question": t.question,
                "finding": t.summary,
                "open_gaps": t.key_gaps,
            }
            for t in self.turns
        ]

    def print_history(self) -> None:
        """Print a clean summary of the full conversation so far."""
        if not self.turns:
            print("No turns yet.")
            return
        print(f"\n{'='*60}")
        print("Conversation history")
        print(f"{'='*60}")
        for t in self.turns:
            print(f"\nQ{t.turn_id+1}: {t.question}")
            print(f"  → {t.summary[:120]}")
            for g in t.key_gaps[:2]:
                print(f"  ⚠ {g[:80]}")
        print()

    def reset(self) -> None:
        """Start a new conversation, keeping the pipeline state."""
        self.turns = []
        print("[conversation] Conversation reset.")
