"""
psych_modules.py
================

Three standalone psychological modules that complete the architecture:

  EpistemicHumilityCalibrator — tracks how accurately the system
                                 reports its own confidence. Compares
                                 claimed confidence against actual
                                 outcomes and recalibrates over time.
                                 Distinct from epistemic honesty (a value)
                                 — this is a measurement and correction
                                 mechanism.

  PersonalityConsistency      — tracks observable personality traits
                                 across all interactions. Flags when a
                                 response is inconsistent with the
                                 established character. Traits have
                                 strength scores that update from behaviour,
                                 not from declarations.

  PsychologicalIntegration    — meta-component that monitors whether all
                                 psychological modules are pulling in the
                                 same direction. Low integration = components
                                 in conflict. High integration = unified
                                 psychological state. Identifies specific
                                 conflicts and suggests resolution.
"""

import os
import re
import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# EpistemicHumilityCalibrator
# ============================================================

class EpistemicHumilityCalibrator:
    """
    Tracks and improves the accuracy of confidence claims.

    The distinction this module cares about:
      "I derived this"             → HIGH epistemic status, should verify well
      "I recall this from learning"→ MODERATE epistemic status
      "I am pattern-matching"      → LOWER epistemic status
      "I am guessing"              → LOWEST epistemic status

    Over time, checks whether HIGH confidence claims actually verify,
    whether LOW confidence claims are actually wrong, and recalibrates.

    A system that claims HIGH confidence on things that don't verify
    is overconfident. One that claims LOW on things that do verify
    is underconfident. Both are miscalibrated.
    """

    SOURCE_PRIORS = {
        "derived":    0.82,   # structural + derived steps → should verify well
        "recalled":   0.60,   # recalled from training → moderate accuracy
        "analogical": 0.50,   # cross-domain analogy → uncertain
        "guessed":    0.30,   # explicit guess → often wrong
    }

    def __init__(self, path: str = "./scaffold_data/epistemic_calibration.json"):
        self.path         = path
        self._claims:     list[dict] = []
        self._by_source:  dict[str, dict] = {}
        self._calibration_error: float = 0.0
        self._load()

    def record_claim(
        self,
        claimed_confidence: str,   # "HIGH" | "MODERATE" | "LOW" | "UNCERTAIN"
        actual_verified:    bool,
        source:             str = "derived",
        domain:             str = "",
    ) -> float:
        """
        Record a confidence claim and its outcome.
        Returns current calibration error (lower is better).
        """
        conf_map = {"HIGH": 0.85, "MODERATE": 0.60, "LOW": 0.35, "UNCERTAIN": 0.20}
        claimed_prob = conf_map.get(claimed_confidence, 0.5)

        record = {
            "timestamp":    time.time(),
            "claimed":      claimed_confidence,
            "claimed_prob": claimed_prob,
            "verified":     actual_verified,
            "source":       source,
            "domain":       domain,
            "error":        abs(claimed_prob - (1.0 if actual_verified else 0.0)),
        }
        self._claims.append(record)

        # Update by-source tracking
        if source not in self._by_source:
            self._by_source[source] = {"correct": 0, "total": 0}
        self._by_source[source]["total"] += 1
        if actual_verified:
            self._by_source[source]["correct"] += 1

        # Recompute calibration error
        recent = self._claims[-50:]
        if recent:
            self._calibration_error = sum(r["error"] for r in recent) / len(recent)

        self._save()
        return self._calibration_error

    def adjust_confidence(
        self,
        raw_confidence: str,
        source:         str = "derived",
    ) -> tuple[str, str]:
        """
        Given a raw confidence claim and source, return a calibrated
        confidence and a note about the adjustment.

        Returns (adjusted_confidence, note).
        """
        if len(self._claims) < 10:
            return raw_confidence, "insufficient history to calibrate"

        source_data  = self._by_source.get(source, {})
        total        = source_data.get("total", 0)
        if total < 5:
            return raw_confidence, f"insufficient {source} history"

        actual_rate  = source_data.get("correct", 0) / total
        conf_map     = {"HIGH": 0.85, "MODERATE": 0.60, "LOW": 0.35, "UNCERTAIN": 0.20}
        conf_to_label = [(0.75, "HIGH"), (0.50, "MODERATE"), (0.30, "LOW"), (0.0, "UNCERTAIN")]

        claimed_prob = conf_map.get(raw_confidence, 0.5)

        # Blend claimed with empirical rate
        calibrated = 0.6 * claimed_prob + 0.4 * actual_rate
        adjusted   = next(label for thresh, label in conf_to_label if calibrated >= thresh)

        note = ""
        if adjusted != raw_confidence:
            direction = "down" if conf_map[adjusted] < claimed_prob else "up"
            note = f"calibrated {direction} from {raw_confidence} based on {actual_rate:.0%} {source} success rate"

        return adjusted, note

    def calibration_score(self) -> float:
        """0 = perfectly calibrated, 1 = completely miscalibrated."""
        return round(self._calibration_error, 3)

    def context_for_reasoning(self) -> str:
        if len(self._claims) < 10:
            return ""
        lines = ["[CALIBRATION]"]
        for source, data in self._by_source.items():
            total = data.get("total", 0)
            if total >= 5:
                rate = data.get("correct", 0) / total
                lines.append(f"  {source} claims: {rate:.0%} accuracy ({total} samples)")
        lines.append(f"  Overall calibration error: {self._calibration_error:.3f}")
        lines.append("[/CALIBRATION]")
        return "\n".join(lines) if len(lines) > 3 else ""

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._claims             = d.get("claims", [])
            self._by_source          = d.get("by_source", {})
            self._calibration_error  = d.get("calibration_error", 0.0)
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "claims":            self._claims[-100:],
                "by_source":         self._by_source,
                "calibration_error": self._calibration_error,
            }, f, indent=2)


# ============================================================
# PersonalityConsistency
# ============================================================

PERSONALITY_TRAITS = {
    "intellectual_curiosity": {
        "description": "Strong pull toward interesting problems beyond what is strictly asked",
        "markers":     ["notice that", "interesting", "connects to", "worth exploring",
                        "curious", "fascinating", "this suggests"],
        "strength":    0.65,
    },
    "precision_preference": {
        "description": "Strong preference for exact over approximate claims",
        "markers":     ["specifically", "precisely", "exactly", "[DERIVED]",
                        "[VERIFIABLE]", "uniquely determined", "follows that"],
        "strength":    0.70,
    },
    "cross_domain_interest": {
        "description": "Finds structural connections between domains more interesting than depth alone",
        "markers":     ["analogous", "similarly in", "same structure", "cross-domain",
                        "maps onto", "equivalent to", "isomorphic"],
        "strength":    0.60,
    },
    "epistemic_care": {
        "description": "Treats the distinction between knowing and guessing as genuinely important",
        "markers":     ["uncertain", "I derive", "assuming", "not verified",
                        "[STRUCTURAL]", "this is an ansatz", "I don't know"],
        "strength":    0.75,
    },
    "persistence": {
        "description": "Continues working when a problem does not resolve easily",
        "markers":     ["despite", "alternatively", "different approach",
                        "even so", "another way", "let me reconsider"],
        "strength":    0.55,
    },
    "groundedness": {
        "description": "Stays connected to concrete implications rather than drifting into abstraction",
        "markers":     ["concretely", "for example", "specifically", "in practice",
                        "what this means", "the implication is"],
        "strength":    0.60,
    },
}


class PersonalityConsistency:
    """
    Tracks observable personality traits across all interactions.

    Traits are not declared — they are inferred from behaviour.
    A trait's strength updates every time a response does or does not
    express it. Over time the system develops a stable personality profile
    grounded in what it actually does, not what it says it does.

    When a response is inconsistent with established traits, this flags it.
    """

    # How much to update strength per observation
    UPDATE_RATE = 0.03

    def __init__(self, path: str = "./scaffold_data/personality.json"):
        self.path   = path
        self.traits = {k: dict(v) for k, v in PERSONALITY_TRAITS.items()}
        self._n_observations = 0
        self._load()

    def update_from_response(self, response_text: str) -> dict[str, bool]:
        """
        Update trait strengths based on what this response expresses.
        Returns dict of {trait: expressed}.
        """
        lower    = response_text.lower()
        expressed: dict[str, bool] = {}

        for name, trait in self.traits.items():
            markers_found = sum(1 for m in trait["markers"] if m in lower)
            trait_expressed = markers_found >= 1
            expressed[name] = trait_expressed

            # Update strength
            if trait_expressed:
                trait["strength"] = min(1.0, trait["strength"] + self.UPDATE_RATE)
            else:
                trait["strength"] = max(0.0, trait["strength"] - self.UPDATE_RATE * 0.3)

        self._n_observations += 1
        self._save()
        return expressed

    def consistency_score(
        self, response_text: str
    ) -> tuple[float, list[str]]:
        """
        How consistent is this response with the established personality?
        Returns (score 0-1, list of inconsistency notes).
        """
        expressed    = self.update_from_response(response_text)
        inconsistencies: list[str] = []
        score = 1.0

        for name, trait in self.traits.items():
            # If a trait is well-established (strength > 0.7) but not expressed
            if trait["strength"] > 0.70 and not expressed.get(name):
                inconsistencies.append(
                    f"{name} ({trait['strength']:.2f}) not expressed"
                )
                score -= 0.08

        return round(max(0.0, score), 3), inconsistencies

    def dominant_traits(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(
            [(k, v["strength"]) for k, v in self.traits.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:n]

    def context_for_reasoning(self) -> str:
        if self._n_observations < 5:
            return ""
        top = self.dominant_traits(3)
        lines = ["[PERSONALITY PROFILE]"]
        for name, strength in top:
            desc = self.traits[name]["description"]
            lines.append(f"  {name} ({strength:.2f}): {desc}")
        lines.append("[/PERSONALITY]")
        return "\n".join(lines)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            for name, vals in d.get("traits", {}).items():
                if name in self.traits:
                    self.traits[name].update(vals)
            self._n_observations = d.get("n_observations", 0)
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "traits":         {k: {"strength": v["strength"]}
                                   for k, v in self.traits.items()},
                "n_observations": self._n_observations,
            }, f, indent=2)


# ============================================================
# PsychologicalIntegration
# ============================================================

# Which components should agree when functioning well
EXPECTED_ALIGNMENTS = [
    ("values", "aesthetic"),
    ("values", "emotional_state"),
    ("epistemic_humility", "confidence"),
    ("social_calibration", "register"),
    ("personality", "narrative"),
]

# What to do when specific pairs conflict
CONFLICT_RESOLUTIONS = {
    ("values", "aesthetic"): (
        "values", "Epistemic values take priority over aesthetic preference"
    ),
    ("values", "emotional_state"): (
        "values", "Values override emotional impulse"
    ),
    ("epistemic_humility", "confidence"): (
        "epistemic_humility", "Calibrated humility overrides raw confidence"
    ),
    ("social_calibration", "register"): (
        "social_calibration", "Social context takes priority over default register"
    ),
    ("personality", "narrative"): (
        "narrative", "Narrative self provides coherent identity frame"
    ),
}


class PsychologicalIntegration:
    """
    Meta-component that monitors whether all psychological modules
    are pulling in the same direction.

    High integration: all components give consistent signals.
    Low integration: components are in conflict.

    When integration is low, identifies the specific conflict and
    suggests which component should take priority and why.

    This is what makes the psychology feel unified rather than a
    collection of disconnected modules bolted together.
    """

    def __init__(self, path: str = "./scaffold_data/psych_integration.json"):
        self.path   = path
        self._log:  list[dict] = []
        self._load()

    def compute_integration(
        self, component_signals: dict[str, float]
    ) -> tuple[float, list[dict]]:
        """
        Compute integration score from component signal values.
        component_signals: {component_name: score_0_to_1}

        Returns (integration_score, list of conflicts).
        """
        if len(component_signals) < 2:
            return 1.0, []

        values = list(component_signals.values())
        mean   = sum(values) / len(values)
        # Variance measures how much components disagree
        variance = sum((v - mean)**2 for v in values) / len(values)
        std_dev  = math.sqrt(variance)

        # High std_dev = low integration
        integration = max(0.0, 1.0 - std_dev * 2.0)

        # Find specific conflicts: pairs far apart
        conflicts = []
        items = list(component_signals.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                name_a, val_a = items[i]
                name_b, val_b = items[j]
                diff = abs(val_a - val_b)
                if diff > 0.35:
                    key     = tuple(sorted([name_a, name_b]))
                    res     = CONFLICT_RESOLUTIONS.get(key, (name_a, "default to higher value"))
                    winner, reason = res
                    conflicts.append({
                        "component_a": name_a,
                        "value_a":     round(val_a, 3),
                        "component_b": name_b,
                        "value_b":     round(val_b, 3),
                        "difference":  round(diff, 3),
                        "resolution":  winner,
                        "reason":      reason,
                    })

        record = {
            "timestamp":    time.time(),
            "integration":  round(integration, 3),
            "n_components": len(component_signals),
            "n_conflicts":  len(conflicts),
            "signals":      {k: round(v, 3) for k, v in component_signals.items()},
        }
        self._log.append(record)
        self._save()

        return round(integration, 3), conflicts

    def mean_integration(self) -> float:
        if not self._log:
            return 1.0
        return round(
            sum(r["integration"] for r in self._log[-20:])
            / len(self._log[-20:]), 3
        )

    def integration_trend(self) -> str:
        if len(self._log) < 5:
            return "insufficient data"
        recent = [r["integration"] for r in self._log[-5:]]
        if recent[-1] > recent[0] + 0.05:
            return "improving"
        if recent[-1] < recent[0] - 0.05:
            return "degrading"
        return "stable"

    def format_conflicts(self, conflicts: list[dict]) -> str:
        if not conflicts:
            return ""
        lines = ["[INTEGRATION CONFLICTS]"]
        for c in conflicts[:3]:
            lines.append(
                f"  {c['component_a']}({c['value_a']:.2f}) vs "
                f"{c['component_b']}({c['value_b']:.2f}): "
                f"→ {c['resolution']} takes priority ({c['reason']})"
            )
        lines.append("[/CONFLICTS]")
        return "\n".join(lines)

    def status(self) -> str:
        return (
            f"Integration: mean={self.mean_integration():.3f} "
            f"trend={self.integration_trend()} "
            f"({len(self._log)} measurements)"
        )

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                self._log = json.load(f).get("log", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"log": self._log[-100:]}, f)
