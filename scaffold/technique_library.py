"""
technique_library.py
====================

Persistent memory of successful reasoning techniques.

Where `obligation_store.py` remembers what *hasn't been solved*, this module
remembers what *has worked*. When a defender branch successfully repairs a
falsified claim, or a particular method unlocks a stuck problem, that pattern
is distilled into a reusable technique and stored. Future runs on similar
questions retrieve and apply relevant techniques as part of branch generation.

Two complementary memories:
  - obligation_store : the open frontier (problems)
  - technique_library : the playbook (solutions that have worked)

The library has three entry points:
  1. record_success(...)        -- save a successful technique manually
  2. extract_techniques_from_run(...) -- ask the LLM to distill techniques
                                          from a completed run
  3. find_relevant_techniques(...) -- retrieve techniques applicable to a
                                       new question, for prompt injection
"""

import os
import re
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional


def _technique_id(name: str, domain: str) -> str:
    raw = f"{domain}::{name.lower().strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _normalise_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().strip())


@dataclass
class Technique:
    """A reusable reasoning move that has worked at least once."""

    technique_id: str
    name: str                # short label (e.g. "promote constant to fn of M")
    description: str         # what the move does
    when_to_use: str         # signal for when it applies (keywords, conditions)
    example_text: str        # concrete example from a successful run
    domain: str

    successful_runs: list[str] = field(default_factory=list)
    failed_runs: list[str] = field(default_factory=list)

    n_uses: int = 0
    n_successes: int = 0
    n_failures: int = 0
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None

    @property
    def success_rate(self) -> Optional[float]:
        if self.n_uses == 0:
            return None
        return self.n_successes / self.n_uses

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success_rate"] = self.success_rate
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Technique":
        d = dict(d)
        d.pop("success_rate", None)
        return cls(**d)


class TechniqueLibrary:
    """Persistent JSON-backed store of successful reasoning techniques."""

    def __init__(self, path: str = "./technique_library.json"):
        self.path = path
        self._data: dict = {"version": 1, "techniques": {}}
        self._load()

    # -------------------- I/O --------------------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not load technique library: {e}")
            print(f"Starting fresh at {self.path}")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    # -------------------- Recording --------------------

    def record_success(
        self,
        name: str,
        description: str,
        when_to_use: str,
        example_text: str,
        domain: str,
        run_id: str,
    ) -> str:
        """Record a successful application of a technique. Idempotent on
        (name, domain): re-recording bumps the success counter."""
        tid = _technique_id(name, domain)
        store = self._data["techniques"]

        if tid in store:
            t = store[tid]
            if run_id not in t["successful_runs"]:
                t["successful_runs"].append(run_id)
            t["n_uses"] = t.get("n_uses", 0) + 1
            t["n_successes"] = t.get("n_successes", 0) + 1
            t["last_used_at"] = time.time()
        else:
            t = Technique(
                technique_id=tid,
                name=name,
                description=description,
                when_to_use=when_to_use,
                example_text=example_text,
                domain=domain,
                successful_runs=[run_id],
                n_uses=1,
                n_successes=1,
                last_used_at=time.time(),
            ).to_dict()
            store[tid] = t

        self.save()
        return tid

    def record_failure(self, technique_id: str, run_id: str,
                       reason: str = "") -> None:
        """Record that a known technique was tried and didn't help."""
        if technique_id not in self._data["techniques"]:
            return
        t = self._data["techniques"][technique_id]
        if run_id not in t["failed_runs"]:
            t["failed_runs"].append(run_id)
        t["n_uses"] = t.get("n_uses", 0) + 1
        t["n_failures"] = t.get("n_failures", 0) + 1
        t["last_used_at"] = time.time()
        if reason:
            t.setdefault("failure_notes", []).append(
                {"run_id": run_id, "reason": reason}
            )
        self.save()

    # -------------------- LLM-driven extraction --------------------

    EXTRACTION_PROMPT = (
        "You are reflecting on a completed reasoning run to extract reusable "
        "techniques for future problems.\n\n"
        "Domain: {domain}\n"
        "Question: {question}\n\n"
        "Successful branches (those that discharged obligations or had high "
        "scores):\n{successful_branches}\n\n"
        "Resolved obligations (problems that got solved):\n{discharged}\n\n"
        "Identify 1-3 GENERALISABLE techniques used in these successful "
        "branches. A technique should be:\n"
        "  - Abstract enough to apply to other problems in this domain\n"
        "  - Specific enough to be actionable when invoked\n"
        "  - Distinct from techniques another reasoner would obviously try\n\n"
        "Output strict JSON:\n"
        "{{\n"
        '  "techniques": [\n'
        '    {{\n'
        '      "name": "short label, 5-10 words",\n'
        '      "description": "what the technique does",\n'
        '      "when_to_use": "specific signal that suggests trying this",\n'
        '      "example_text": "1-2 sentence concrete example from this run"\n'
        '    }}\n'
        "  ]\n"
        "}}\n"
        "If nothing in this run is genuinely reusable, return "
        '{{"techniques": []}}.\n'
    )

    def extract_techniques_from_run(
        self,
        run_id: str,
        domain: str,
        question: str,
        branches: list[dict],
        resolution: dict,
        llm_chat_fn: Callable,
        score_attr: str = "total_score",
    ) -> list[str]:
        """
        Ask the LLM to distill reusable techniques from a successful run.
        Returns the list of new technique_ids recorded.
        """
        # Identify successful branches: those that discharged any obligation,
        # or those with the highest scores.
        discharge_matrix = resolution.get("discharge_matrix", {})
        branches_that_discharged = set()
        for branches_list in discharge_matrix.values():
            for b in branches_list:
                branches_that_discharged.add(b)

        successful = [
            b for b in branches
            if b.get("name") in branches_that_discharged
        ]

        if not successful:
            scored = [b for b in branches if score_attr in b]
            scored.sort(key=lambda b: b.get(score_attr, 0), reverse=True)
            successful = scored[: max(1, len(scored) // 3)]

        if not successful:
            return []

        discharged_obligations = [
            o for o in resolution.get("obligations", [])
            if o.get("status") == "discharged"
        ]

        prompt = self.EXTRACTION_PROMPT.format(
            domain=domain,
            question=question,
            successful_branches=json.dumps(
                [{"name": b.get("name"), "method": b.get("method"),
                  "steps": b.get("steps"),
                  "result": b.get("candidate_result")}
                 for b in successful],
                indent=2,
            ),
            discharged=json.dumps(
                [o.get("text") for o in discharged_obligations],
                indent=2,
            ),
        )

        try:
            response = llm_chat_fn(
                "You distill reusable reasoning techniques. Return only "
                "valid JSON.",
                prompt,
            )
        except Exception as e:
            print(f"Technique extraction failed: {e}")
            return []

        techniques = self._parse_extraction_response(response)
        new_ids = []
        for t in techniques:
            tid = self.record_success(
                name=t.get("name", "unnamed"),
                description=t.get("description", ""),
                when_to_use=t.get("when_to_use", ""),
                example_text=t.get("example_text", ""),
                domain=domain,
                run_id=run_id,
            )
            new_ids.append(tid)
        return new_ids

    @staticmethod
    def _parse_extraction_response(text: str) -> list[dict]:
        try:
            obj = json.loads(text)
            return obj.get("techniques", [])
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return []
        try:
            obj = json.loads(match.group(0))
            return obj.get("techniques", [])
        except json.JSONDecodeError:
            return []

    # -------------------- Retrieval --------------------

    def find_relevant_techniques(
        self,
        question: str,
        domain: str,
        top_n: int = 3,
        min_success_rate: float = 0.0,
    ) -> list[Technique]:
        """
        Retrieve techniques applicable to a new question. Uses a simple
        keyword-overlap score between the question and each technique's
        when_to_use field. Restricts to the matching domain.
        """
        q_tokens = self._tokenise(question)
        if not q_tokens:
            return []

        scored = []
        for t_dict in self._data["techniques"].values():
            if t_dict.get("domain") != domain:
                continue
            if t_dict.get("n_uses", 0) > 0:
                rate = t_dict.get("n_successes", 0) / t_dict["n_uses"]
            else:
                rate = 1.0
            if rate < min_success_rate:
                continue

            text = " ".join([
                t_dict.get("when_to_use", ""),
                t_dict.get("description", ""),
                t_dict.get("name", ""),
            ])
            t_tokens = self._tokenise(text)
            if not t_tokens:
                continue

            overlap = len(q_tokens & t_tokens)
            # Weight by success rate so well-tested techniques rank higher.
            score = overlap * (0.5 + 0.5 * rate)
            if score > 0:
                scored.append((score, Technique.from_dict(t_dict)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_n]]

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        STOPWORDS = {
            "the", "a", "an", "is", "are", "be", "to", "of", "and", "or",
            "for", "in", "on", "at", "by", "with", "from", "as", "this",
            "that", "it", "do", "does", "can", "will", "should", "could",
            "would", "any", "all", "some", "what", "when", "where", "how",
            "why", "which", "if", "then", "than", "but",
        }
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]+", text.lower())
        return {t for t in tokens if t not in STOPWORDS and len(t) > 2}

    # -------------------- Prompt formatting --------------------

    def format_for_prompt(self, techniques: list[Technique]) -> str:
        """Format retrieved techniques for injection into the
        branch-generation prompt."""
        if not techniques:
            return ""
        lines = [
            "Techniques that have worked on similar problems before "
            "(consider whether any apply here):"
        ]
        for t in techniques:
            sr = t.success_rate
            sr_str = f"{sr:.0%}" if sr is not None else "untested"
            lines.append(f"  - **{t.name}** (success rate: {sr_str})")
            lines.append(f"      What it does: {t.description}")
            lines.append(f"      When it applies: {t.when_to_use}")
            if t.example_text:
                lines.append(f"      Example: {t.example_text}")
        return "\n".join(lines)

    # -------------------- Summary --------------------

    def summary(self) -> dict:
        techs = self._data["techniques"]
        if not techs:
            return {"n_techniques": 0}

        rates = []
        for t in techs.values():
            n = t.get("n_uses", 0)
            if n > 0:
                rates.append(t.get("n_successes", 0) / n)
        return {
            "n_techniques": len(techs),
            "domains": sorted({t.get("domain", "") for t in techs.values()}),
            "mean_success_rate": (sum(rates) / len(rates)) if rates else None,
            "most_used": sorted(
                [{"name": t["name"], "uses": t.get("n_uses", 0),
                  "success_rate": (
                      t.get("n_successes", 0) / t["n_uses"]
                      if t.get("n_uses", 0) > 0 else None)}
                 for t in techs.values()],
                key=lambda x: x["uses"], reverse=True,
            )[:5],
        }
