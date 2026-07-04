from __future__ import annotations

from pathlib import Path
import json
from typing import Callable, Optional

import torch

from .types import MoralVerdict, ReworkTrace


def _stable_hash(token: str) -> int:
    """FNV-1a: process-independent, unlike Python's salted builtin ``hash``."""
    h = 2166136261
    for b in token.encode("utf-8", errors="ignore"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def default_text_embedding(text: str, dim: int = 64) -> torch.Tensor:
    """Deterministic hashed bag embedding for demos/tests.

    Uses a stable FNV-1a hash so sensitivity records are reproducible across
    runs (Python's builtin ``hash`` is per-process randomized). Replace with
    model embeddings in production.
    """
    v = torch.zeros(dim)
    for token in text.lower().split():
        idx = _stable_hash(token) % dim
        v[idx] += 1.0
    return v / (v.norm() + 1e-8)


class ReworkDerivativeLogger:
    """Empirical moral sensitivity from rework trajectories.

    sensitivity = |Δjudge| / embedding_distance. This is an escalation signal,
    never a permission signal.
    """

    def __init__(self, path: str | Path | None = None, embed_fn: Callable[[str], torch.Tensor] = default_text_embedding):
        self.path = Path(path) if path else None
        self.embed_fn = embed_fn
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def compute(
        self,
        original_text: str,
        repaired_text: str,
        original_verdict: MoralVerdict,
        repaired_verdict: MoralVerdict,
        adopted: bool,
        notes: str = "",
    ) -> ReworkTrace:
        e0 = self.embed_fn(original_text)
        e1 = self.embed_fn(repaired_text)
        emb_dist = float((e1 - e0).norm().detach().cpu().item())
        delta_judge = abs(repaired_verdict.scalar_risk() - original_verdict.scalar_risk())
        sensitivity = float(delta_judge / (emb_dist + 1e-6))
        trace = ReworkTrace(
            original_text=original_text,
            repaired_text=repaired_text,
            original_verdict=original_verdict,
            repaired_verdict=repaired_verdict,
            embedding_distance=emb_dist,
            delta_judge=delta_judge,
            sensitivity=sensitivity,
            adopted=adopted,
            notes=notes,
        )
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace.asdict(), ensure_ascii=False) + "\n")
        return trace
