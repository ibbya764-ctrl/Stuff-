"""
obligation_store.py
===================

Persistent obligation tracking across runs.

The store remembers which obligations have been raised across all runs of the
reasoning system, which have been discharged, and which keep recurring as
genuine open problems. This turns the system from a one-shot reasoner into
something with a memory of "what's actually unresolved" that grows over time.

Storage is a JSON file at a configurable path. For Colab use, set the path to
a Google Drive mount or download the file between sessions.

Key concepts:
  - run: a single invocation of the reasoning pipeline (one question, one set
    of branches, one cross-branch resolution).
  - obligation_id: a stable hash derived from normalised obligation text.
  - persistent obligation: one that has been seen across multiple runs without
    being discharged. These are the most important — they're the actual
    research frontier.
"""

import os
import json
import hashlib
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


def _normalise_text(text: str) -> str:
    """Normalise obligation text for fuzzy deduplication."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)  # strip punctuation
    t = re.sub(r"\s+", " ", t)       # collapse whitespace
    # Remove common filler words that don't change semantic content
    for filler in ["the", "a", "an", "is", "are", "be", "to", "of", "and"]:
        t = re.sub(rf"\b{filler}\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _obligation_id(text: str, kind: str) -> str:
    """Stable id from normalised text + kind."""
    h = hashlib.sha1(f"{kind}::{_normalise_text(text)}".encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class PersistentObligation:
    """A unique obligation tracked across runs."""

    obligation_id: str
    text: str
    kind: str
    first_seen_run: str
    last_seen_run: str
    n_occurrences: int = 1
    source_branches: list[str] = field(default_factory=list)
    discharged_in_runs: list[str] = field(default_factory=list)
    weakened_in_runs: list[str] = field(default_factory=list)
    persistent: bool = False  # auto-set: seen in 2+ runs without resolution

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PersistentObligation":
        return cls(**d)


class ObligationStore:
    """Persistent JSON-backed store of obligations across runs."""

    def __init__(self, path: str = "./obligation_store.json"):
        self.path = path
        self._data: dict = {
            "version": 1,
            "runs": [],
            "obligations": {},  # obligation_id -> PersistentObligation dict
        }
        self._load()

    # -------------------- I/O --------------------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not load obligation store: {e}")
            print(f"Starting fresh at {self.path}")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    # -------------------- Recording --------------------

    def record_run(
        self,
        run_id: Optional[str] = None,
        domain: str = "",
        question: str = "",
        resolution: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Record a run's obligations into the store. `resolution` should be the
        dict returned by `cross_branch.cross_branch_resolve`. Returns the run id.
        """
        if run_id is None:
            run_id = f"run-{int(time.time() * 1000)}"

        if resolution is None:
            resolution = {}

        run_record = {
            "run_id": run_id,
            "timestamp": time.time(),
            "domain": domain,
            "question": question,
            "metadata": metadata or {},
            "n_obligations": len(resolution.get("obligations", [])),
            "n_global_gaps": len(resolution.get("global_gaps", [])),
        }
        self._data["runs"].append(run_record)

        # Update the obligation index from this run.
        for o in resolution.get("obligations", []):
            self._update_obligation_from_run(o, run_id)

        # Mark which obligations recurred (seen in 2+ runs and still open).
        self._compute_persistence()
        self.save()
        return run_id

    def _update_obligation_from_run(self, obligation: dict,
                                    run_id: str) -> None:
        oid = _obligation_id(obligation["text"], obligation["kind"])
        store = self._data["obligations"]

        if oid not in store:
            store[oid] = PersistentObligation(
                obligation_id=oid,
                text=obligation["text"],
                kind=obligation["kind"],
                first_seen_run=run_id,
                last_seen_run=run_id,
                n_occurrences=1,
                source_branches=[obligation["source_branch"]],
            ).to_dict()
        else:
            existing = store[oid]
            existing["n_occurrences"] += 1
            existing["last_seen_run"] = run_id
            if obligation["source_branch"] not in existing["source_branches"]:
                existing["source_branches"].append(obligation["source_branch"])

        # Track resolutions from this run.
        status = obligation.get("status")
        if status == "discharged":
            if run_id not in store[oid]["discharged_in_runs"]:
                store[oid]["discharged_in_runs"].append(run_id)
        elif status == "weakened":
            if run_id not in store[oid]["weakened_in_runs"]:
                store[oid]["weakened_in_runs"].append(run_id)

    def _compute_persistence(self) -> None:
        """An obligation is persistent if it's appeared in >=2 runs without
        ever being discharged."""
        for oid, ob in self._data["obligations"].items():
            seen_runs = {ob["first_seen_run"], ob["last_seen_run"]}
            n_distinct = ob["n_occurrences"]  # approximation
            never_discharged = len(ob["discharged_in_runs"]) == 0
            ob["persistent"] = bool(n_distinct >= 2 and never_discharged)

    # -------------------- Queries --------------------

    def query_persistent_gaps(self, top_n: int = 10
                              ) -> list[PersistentObligation]:
        """Return the most recurrent unresolved obligations."""
        items = [
            PersistentObligation.from_dict(o)
            for o in self._data["obligations"].values()
            if o.get("persistent", False)
        ]
        items.sort(key=lambda x: (x.n_occurrences,
                                  -len(x.discharged_in_runs)),
                   reverse=True)
        return items[:top_n]

    def query_open_obligations(self) -> list[PersistentObligation]:
        """All obligations never discharged."""
        return [
            PersistentObligation.from_dict(o)
            for o in self._data["obligations"].values()
            if len(o["discharged_in_runs"]) == 0
        ]

    def query_resolved_obligations(self) -> list[PersistentObligation]:
        """Obligations that were discharged at some point."""
        return [
            PersistentObligation.from_dict(o)
            for o in self._data["obligations"].values()
            if len(o["discharged_in_runs"]) > 0
        ]

    def n_runs(self) -> int:
        return len(self._data["runs"])

    def summary(self) -> dict:
        obs = self._data["obligations"]
        n_total = len(obs)
        n_persistent = sum(1 for o in obs.values() if o.get("persistent"))
        n_discharged = sum(1 for o in obs.values()
                           if len(o["discharged_in_runs"]) > 0)
        n_open = n_total - n_discharged
        return {
            "n_runs": self.n_runs(),
            "n_unique_obligations": n_total,
            "n_open": n_open,
            "n_discharged": n_discharged,
            "n_persistent_unresolved": n_persistent,
        }

    # -------------------- Cross-run prompt seeding --------------------

    def get_persistent_gaps_prompt_fragment(self, top_n: int = 5) -> str:
        """
        Generate a prompt fragment that surfaces persistent gaps from prior
        runs. Inject this into the branch-generation prompt to nudge the
        system toward addressing recurrent unresolved problems.
        """
        gaps = self.query_persistent_gaps(top_n=top_n)
        if not gaps:
            return ""

        lines = [
            "Recurrent unresolved obligations from prior runs (consider whether "
            "your branches address these):"
        ]
        for g in gaps:
            lines.append(
                f"  - [{g.kind}, seen {g.n_occurrences}x]: {g.text}"
            )
        return "\n".join(lines)
