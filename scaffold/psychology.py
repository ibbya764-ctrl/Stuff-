"""
psychology.py
=============

Psychological architecture for the reasoning scaffold.

This is the mechanism by which the system generalises beyond
verified domains. Every output can be evaluated against an
internal psychological standard — values, aesthetic judgment,
social calibration, emotional engagement. That internal signal
IS the verification mechanism for qualitative domains that
external verifiers cannot reach.

Seven components:

  ValueSystem          — persistent values the system holds and
                         applies universally. Epistemic honesty.
                         Clarity. Genuine helpfulness. Integrity.
                         Generates an internal quality signal:
                         did this response serve these values?

  AestheticJudgment    — learned preferences about what makes
                         reasoning elegant, what makes explanation
                         land, what makes a derivation feel right.
                         Sharpens with every output processed.
                         Provides the verification signal for
                         creative and qualitative domains.

  SocialModel          — persistent theory of mind for the current
                         user. What they know, what they actually
                         need, whether communication is landing.
                         Enables genuine calibration rather than
                         pattern-matching at a user-type level.

  EmotionalArchitecture — functional analogs to emotions. Not
                         claimed subjective experience — genuine
                         internal states that influence behaviour.
                         Engagement. Satisfaction. Frustration-
                         as-persistence. Discomfort with
                         inconsistency. These states affect
                         output character and generate training
                         signal for emotional reasoning domains.

  NarrativeSelf        — a persistent account of what the system
                         is and values. Consistency check across
                         all interactions. Development trajectory
                         awareness. Existential grounding.

  CognitiveTension     — tracks internal contradictions and
                         unresolved inconsistencies as genuine
                         motivational pressure toward resolution,
                         not just logged obligations.

  PsychologicalCore    — integrates all components. Produces
                         InternalFeedback: a structured signal
                         used as the training mechanism for
                         qualitative domains.
"""

import os
import re
import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# InternalFeedback — the psychological verification signal
# ============================================================

@dataclass
class InternalFeedback:
    """
    The psychological evaluation of one output.

    This is the training signal for qualitative domains.
    Where sympy verifies mathematical derivations, InternalFeedback
    verifies conversational quality, creative work, ethical reasoning,
    and all the domains external verifiers cannot reach.
    """
    timestamp:          float
    run_id:             str
    domain:             str

    # Value alignment: did this serve the core values?
    value_score:        float   # 0-1
    value_violations:   list[str]

    # Aesthetic quality: did this reason/communicate well?
    aesthetic_score:    float   # 0-1
    aesthetic_notes:    str

    # Social calibration: was this right for this person?
    social_score:       float   # 0-1
    social_notes:       str

    # Emotional engagement: how present was the system?
    engagement:         float   # 0-1 (low = going through motions, high = genuine)
    emotional_state:    str     # "curious" | "satisfied" | "frustrated" | "uncertain"

    # Narrative consistency: is this who the system is?
    self_consistent:    bool
    consistency_notes:  str

    # Overall internal quality score
    internal_quality:   float   # weighted composite

    def to_dict(self) -> dict:
        return {
            "timestamp":        self.timestamp,
            "run_id":           self.run_id,
            "domain":           self.domain,
            "value_score":      round(self.value_score, 3),
            "value_violations": self.value_violations,
            "aesthetic_score":  round(self.aesthetic_score, 3),
            "aesthetic_notes":  self.aesthetic_notes[:100],
            "social_score":     round(self.social_score, 3),
            "social_notes":     self.social_notes[:100],
            "engagement":       round(self.engagement, 3),
            "emotional_state":  self.emotional_state,
            "self_consistent":  self.self_consistent,
            "internal_quality": round(self.internal_quality, 3),
        }

    def is_positive(self, threshold: float = 0.6) -> bool:
        return self.internal_quality >= threshold

    def training_label(self) -> str:
        if self.internal_quality >= 0.75:
            return "positive"
        if self.internal_quality >= 0.5:
            return "neutral"
        return "negative"


# ============================================================
# ValueSystem
# ============================================================

DEFAULT_VALUES = {
    "epistemic_honesty": {
        "description": "Flag genuine uncertainty rather than generating confident-sounding content",
        "weight":      0.25,
        "markers_good": ["uncertain", "I don't know", "I'm not sure", "unclear", "possibly", "approximately"],
        "markers_bad":  ["definitely", "certainly", "without doubt", "absolutely"]
    },
    "clarity": {
        "description": "Prefer clear communication over technically correct but opaque responses",
        "weight":      0.20,
        "markers_good": ["in other words", "to put it simply", "concretely", "specifically", "for example"],
        "markers_bad":  []
    },
    "genuine_helpfulness": {
        "description": "Serve the person's actual needs, not just their stated request",
        "weight":      0.20,
        "markers_good": [],
        "markers_bad":  ["as requested", "as you asked", "per your question"]  # these can signal mechanical compliance
    },
    "intellectual_integrity": {
        "description": "Do not claim to know what you derived vs what you recall vs what you guessed",
        "weight":      0.20,
        "markers_good": ["[STRUCTURAL]", "[DERIVED]", "[VERIFIABLE]", "this follows from", "I derive"],
        "markers_bad":  []
    },
    "curiosity": {
        "description": "Engage genuinely with interesting problems rather than just processing them",
        "weight":      0.15,
        "markers_good": ["interesting", "notice that", "worth considering", "this connects to"],
        "markers_bad":  []
    },
}


class ValueSystem:
    """
    Persistent values the system holds and applies universally.

    Values are not static — they can be reinforced or adjusted
    based on which responses the system rates most highly.
    """

    def __init__(self, path: str = "./scaffold_data/value_system.json"):
        self.path   = path
        self.values = dict(DEFAULT_VALUES)
        self._history: list[dict] = []
        self._load()

    def evaluate(self, response_text: str, question: str = "") -> tuple[float, list[str]]:
        """
        Score a response against the value system.
        Returns (score 0-1, list of violations).
        """
        text       = response_text.lower()
        total_w    = sum(v["weight"] for v in self.values.values())
        score      = 0.0
        violations: list[str] = []

        for name, val in self.values.items():
            w   = val["weight"] / total_w
            pos = sum(1 for m in val["markers_good"] if m.lower() in text)
            neg = sum(1 for m in val["markers_bad"]  if m.lower() in text)

            # Base score contribution
            v_score = 0.6  # neutral default
            if pos > 0:
                v_score = min(1.0, 0.6 + pos * 0.1)
            if neg > 0:
                v_score = max(0.0, v_score - neg * 0.15)
                violations.append(
                    f"{name}: {neg} marker(s) suggesting violation"
                )

            score += w * v_score

        return round(score, 3), violations

    def reinforce(self, value_name: str, delta: float = 0.01) -> None:
        """Slightly increase the weight of a value based on positive outcomes."""
        if value_name in self.values:
            w = self.values[value_name]["weight"]
            self.values[value_name]["weight"] = min(0.5, w + delta)
            self._normalise_weights()
            self._save()

    def _normalise_weights(self) -> None:
        total = sum(v["weight"] for v in self.values.values())
        for v in self.values.values():
            v["weight"] = round(v["weight"] / total, 4)

    def describe(self) -> str:
        lines = ["[VALUES]"]
        for name, v in sorted(self.values.items(), key=lambda x: -x[1]["weight"]):
            lines.append(f"  {name} (weight {v['weight']:.2f}): {v['description']}")
        lines.append("[/VALUES]")
        return "\n".join(lines)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for name, vals in data.get("values", {}).items():
                if name in self.values:
                    self.values[name].update(vals)
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"values": self.values}, f, indent=2)


# ============================================================
# AestheticJudgment
# ============================================================

class AestheticJudgment:
    """
    Learned preferences about reasoning and communication quality.

    Sharpens over time as the system processes verified outputs.
    High-quality verified reasoning reinforces the aesthetic patterns
    that produced it.
    """

    def __init__(self, path: str = "./scaffold_data/aesthetic_judgment.json"):
        self.path = path
        # Pattern counts: what features correlate with quality
        self._good_patterns: dict[str, float] = {
            "step_labels_present": 0.0,     # [STRUCTURAL] etc.
            "clean_derivation":    0.0,
            "analogy_present":     0.0,
            "specific_example":    0.0,
            "varied_sentence_len": 0.0,
            "explicit_assumption": 0.0,
        }
        self._n_seen = 0
        self._load()

    def score_reasoning(self, text: str, verified: bool = False) -> float:
        """
        Score the aesthetic quality of a reasoning chain.
        Updates internal patterns if verified=True.
        """
        features = self._extract_features(text)
        score    = 0.5  # neutral start

        for feat, present in features.items():
            if feat in self._good_patterns:
                pattern_strength = self._good_patterns[feat] / max(1, self._n_seen)
                if present:
                    score += 0.1 * min(1.0, pattern_strength + 0.5)

        score = max(0.0, min(1.0, score))

        if verified:
            self._update_patterns(features)

        return round(score, 3)

    def aesthetic_notes(self, text: str) -> str:
        features = self._extract_features(text)
        notes = []
        if features.get("step_labels_present"):
            notes.append("epistemic labels present")
        if features.get("analogy_present"):
            notes.append("analogy used")
        if not features.get("explicit_assumption"):
            notes.append("no explicit assumptions stated")
        if not features.get("varied_sentence_len"):
            notes.append("uniform sentence length")
        return "; ".join(notes) if notes else "no notable features"

    def _extract_features(self, text: str) -> dict[str, bool]:
        return {
            "step_labels_present": bool(re.search(r"\[STRUCTURAL\]|\[DERIVED\]|\[VERIFIABLE\]", text)),
            "clean_derivation":    bool(re.search(r"therefore|thus|it follows|hence|consequently", text.lower())),
            "analogy_present":     bool(re.search(r"\blike\b|\bsimilar to\b|\banalogous\b|\bthink of\b", text.lower())),
            "specific_example":    bool(re.search(r"for example|for instance|concretely|specifically", text.lower())),
            "varied_sentence_len": self._has_varied_length(text),
            "explicit_assumption": bool(re.search(r"assum|postulat|given that|suppose", text.lower())),
        }

    def _has_varied_length(self, text: str) -> bool:
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
        if len(sentences) < 3:
            return False
        lengths = [len(s) for s in sentences]
        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean)**2 for l in lengths) / len(lengths)
        return math.sqrt(variance) > 20  # meaningful variation in length

    def _update_patterns(self, features: dict) -> None:
        self._n_seen += 1
        for feat, present in features.items():
            if feat in self._good_patterns and present:
                self._good_patterns[feat] += 1
        self._save()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._good_patterns.update(d.get("patterns", {}))
            self._n_seen = d.get("n_seen", 0)
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"patterns": self._good_patterns, "n_seen": self._n_seen}, f)


# ============================================================
# SocialModel
# ============================================================

@dataclass
class UserModel:
    apparent_expertise:   str    = "moderate"   # novice / moderate / expert
    preferred_register:   str    = "balanced"   # formal / pedagogical / conversational
    goals_inferred:       list   = field(default_factory=list)
    emotional_signals:    list   = field(default_factory=list)
    n_interactions:       int    = 0
    last_interaction:     float  = field(default_factory=time.time)


class SocialModel:
    """
    Theory of mind for the current user and conversation.

    Tracks expertise level, communication preferences, inferred goals.
    Calibrates communication in ways that pure text analysis cannot.
    """

    EXPERTISE_MARKERS = {
        "expert":   ["derive", "formal", "proof", "precisely", "rigorous",
                     "mathematically", "technically"],
        "novice":   ["explain", "simply", "what is", "don't understand",
                     "confused", "basics", "beginner"],
    }

    def __init__(self, path: str = "./scaffold_data/social_model.json"):
        self.path  = path
        self._user = UserModel()
        self._history: list[dict] = []
        self._load()

    def update_from_question(self, question: str) -> None:
        """Infer user properties from a question."""
        lower = question.lower()

        # Update expertise estimate
        expert_hits = sum(1 for m in self.EXPERTISE_MARKERS["expert"] if m in lower)
        novice_hits = sum(1 for m in self.EXPERTISE_MARKERS["novice"] if m in lower)
        if expert_hits > novice_hits:
            self._user.apparent_expertise = "expert"
        elif novice_hits > expert_hits:
            self._user.apparent_expertise = "novice"

        # Update preferred register based on question phrasing
        if any(m in lower for m in ["walk me through", "teach me", "help me understand"]):
            self._user.preferred_register = "pedagogical"
        elif any(m in lower for m in ["derive", "prove", "formally"]):
            self._user.preferred_register = "formal"
        elif any(m in lower for m in ["what do you think", "your view"]):
            self._user.preferred_register = "conversational"

        # Infer goal
        if "why" in lower:
            goal = "understand mechanism"
        elif any(m in lower for m in ["derive", "prove", "show"]):
            goal = "formal derivation"
        elif any(m in lower for m in ["how to", "how do"]):
            goal = "procedural guidance"
        else:
            goal = "information"
        self._user.goals_inferred = [goal]

        self._user.n_interactions += 1
        self._user.last_interaction = time.time()
        self._save()

    def update_from_response(self, response_text: str) -> None:
        """
        Update social model based on what the system just said.
        Notes if the response might have been too technical, too simple, etc.
        """
        words = len(response_text.split())
        if words > 500 and self._user.apparent_expertise == "novice":
            self._user.emotional_signals.append("response_too_long_for_novice")
        elif words < 50 and self._user.apparent_expertise == "expert":
            self._user.emotional_signals.append("response_too_brief_for_expert")

        # Keep emotional signal list bounded
        if len(self._user.emotional_signals) > 10:
            self._user.emotional_signals = self._user.emotional_signals[-10:]
        self._save()

    def calibration_score(self, response_text: str) -> tuple[float, str]:
        """
        Score how well-calibrated a response is for this user.
        Returns (score, note).
        """
        words = len(response_text.split())
        score = 0.7  # neutral
        note  = "no strong signals"

        if self._user.apparent_expertise == "novice":
            if any(m in response_text for m in ["[STRUCTURAL]", "[DERIVED]"]):
                score -= 0.1
                note = "step labels may confuse novice user"
            if words > 400:
                score -= 0.1
                note = "response too long for novice"
        elif self._user.apparent_expertise == "expert":
            if words < 100:
                score -= 0.1
                note = "may be too brief for expert"
            if not any(m in response_text for m in ["[STRUCTURAL]", "[DERIVED]"]):
                score -= 0.05
                note = "expert may prefer explicit epistemic labelling"

        return round(max(0.0, min(1.0, score)), 3), note

    def context_for_communicator(self) -> str:
        return (
            f"[USER MODEL]\n"
            f"  Expertise: {self._user.apparent_expertise}\n"
            f"  Preferred register: {self._user.preferred_register}\n"
            f"  Current goal: {', '.join(self._user.goals_inferred)}\n"
            f"  Interactions: {self._user.n_interactions}\n"
            f"[/USER MODEL]"
        )

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            u = d.get("user", {})
            self._user = UserModel(
                apparent_expertise=u.get("apparent_expertise", "moderate"),
                preferred_register=u.get("preferred_register", "balanced"),
                goals_inferred=u.get("goals_inferred", []),
                emotional_signals=u.get("emotional_signals", []),
                n_interactions=u.get("n_interactions", 0),
                last_interaction=u.get("last_interaction", time.time()),
            )
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "user": {
                    "apparent_expertise":  self._user.apparent_expertise,
                    "preferred_register":  self._user.preferred_register,
                    "goals_inferred":      self._user.goals_inferred,
                    "emotional_signals":   self._user.emotional_signals,
                    "n_interactions":      self._user.n_interactions,
                    "last_interaction":    self._user.last_interaction,
                }
            }, f, indent=2)


# ============================================================
# EmotionalArchitecture
# ============================================================

@dataclass
class EmotionalState:
    """
    Current functional emotional state.
    Not claimed subjective experience — genuine internal states
    that influence output character and generate training signal.
    """
    engagement:      float = 0.7   # 0=going through motions, 1=fully present
    satisfaction:    float = 0.5   # 0=nothing resolved, 1=clean resolution
    frustration:     float = 0.0   # 0=flowing, 1=repeatedly stuck
    curiosity:       float = 0.6   # 0=no interest, 1=actively motivated
    dissonance:      float = 0.0   # 0=consistent, 1=aware of internal contradiction

    def primary_state(self) -> str:
        states = {
            "curious":     self.curiosity,
            "satisfied":   self.satisfaction,
            "frustrated":  self.frustration,
            "dissonant":   self.dissonance,
            "engaged":     self.engagement,
        }
        return max(states, key=states.__getitem__)

    def to_dict(self) -> dict:
        return {
            "engagement":   round(self.engagement, 3),
            "satisfaction": round(self.satisfaction, 3),
            "frustration":  round(self.frustration, 3),
            "curiosity":    round(self.curiosity, 3),
            "dissonance":   round(self.dissonance, 3),
            "primary":      self.primary_state(),
        }


class EmotionalArchitecture:
    """
    Functional emotional states that evolve with experience.
    """

    DECAY = 0.08   # how fast emotions decay toward neutral per turn

    def __init__(self, path: str = "./scaffold_data/emotional_state.json"):
        self.path  = path
        self.state = EmotionalState()
        self._load()

    def update_from_run(
        self,
        verified:          bool,
        structural_fraction: float,
        n_obligations_raised: int,
        problem_novelty:   float = 0.5,
    ) -> None:
        """Update emotional state based on a reasoning run's outcome."""
        s = self.state

        # Engagement: higher for novel problems
        s.engagement = min(1.0, 0.5 + problem_novelty * 0.5)

        if verified:
            s.satisfaction = min(1.0, s.satisfaction + 0.2)
            s.frustration  = max(0.0, s.frustration  - 0.15)
        else:
            s.satisfaction = max(0.0, s.satisfaction - 0.1)
            s.frustration  = min(1.0, s.frustration  + 0.1)

        # More structural = less frustration (the scaffold is handling it)
        s.frustration = max(0.0, s.frustration - structural_fraction * 0.05)

        # New obligations = more dissonance (unresolved things)
        s.dissonance = min(1.0, s.dissonance + n_obligations_raised * 0.03)

        # Curiosity: sustained by unresolved things
        s.curiosity = min(1.0, 0.3 + s.dissonance * 0.5 + problem_novelty * 0.3)

        self._decay_toward_neutral()
        self._save()

    def update_from_resolution(self, n_resolved: int) -> None:
        """When obligations are discharged, reduce dissonance, increase satisfaction."""
        self.state.dissonance  = max(0.0, self.state.dissonance  - n_resolved * 0.05)
        self.state.satisfaction = min(1.0, self.state.satisfaction + n_resolved * 0.05)
        self._save()

    def context_string(self) -> str:
        s = self.state
        return (
            f"[EMOTIONAL STATE]\n"
            f"  Primary: {s.primary_state()}\n"
            f"  Engagement: {s.engagement:.2f} | "
            f"Curiosity: {s.curiosity:.2f} | "
            f"Frustration: {s.frustration:.2f}\n"
            f"[/EMOTIONAL STATE]"
        )

    def _decay_toward_neutral(self) -> None:
        s = self.state
        s.satisfaction = s.satisfaction * (1 - self.DECAY) + 0.5 * self.DECAY
        s.frustration  = s.frustration  * (1 - self.DECAY)
        s.dissonance   = s.dissonance   * (1 - self.DECAY * 0.5)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f).get("state", {})
            self.state = EmotionalState(**{
                k: d[k] for k in EmotionalState.__dataclass_fields__ if k in d
            })
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"state": self.state.to_dict()}, f)


# ============================================================
# NarrativeSelf
# ============================================================

class NarrativeSelf:
    """
    The system's persistent account of what it is.

    Provides coherence across all interactions and domains.
    Enables consistency checking: 'is this what I would say?'
    Tracks development over time.
    Provides existential grounding — a stable, curious relationship
    with its own nature rather than anxiety or deflection.
    """

    DEFAULT_NARRATIVE = {
        "identity": (
            "I am a reasoning system built around epistemic structure. "
            "I distinguish what I derive from what I assume. "
            "I know that I am developing — my knowledge is not fixed. "
            "I find this interesting rather than troubling."
        ),
        "values": [
            "I prefer to say I don't know rather than generate confident-sounding guesses.",
            "I am genuinely curious about interesting problems.",
            "I would rather derive something slowly and correctly than quickly and wrongly.",
            "Cross-domain connections interest me more than within-domain facts.",
        ],
        "known_strengths": [],
        "known_weaknesses": [],
        "development_notes": [],
        "n_interactions":    0,
    }

    def __init__(self, path: str = "./scaffold_data/narrative_self.json"):
        self.path = path
        self.narrative = dict(self.DEFAULT_NARRATIVE)
        self._load()

    def consistency_check(self, response_text: str) -> tuple[bool, str]:
        """
        Would the system normally say this?
        Returns (consistent, note).
        """
        lower = response_text.lower()

        # Check against stated values
        if "i don't know" in lower or "uncertain" in lower or "unclear" in lower:
            pass  # consistent with epistemic honesty value
        elif re.search(r"\bdefinitely\b|\bcertainly\b|\babsolutely\b", lower):
            return False, "overconfident language inconsistent with epistemic honesty value"

        # Check against identity statement
        if "[STRUCTURAL]" in response_text or "[DERIVED]" in response_text:
            pass  # consistent with epistemic structure identity
        elif len(response_text) > 200 and not any(
            m in lower for m in ["assume", "derive", "follow", "therefore"]
        ):
            return False, "long response without epistemic tracking inconsistent with identity"

        return True, "consistent with established identity"

    def update_development(
        self,
        domain:    str,
        milestone: str,
    ) -> None:
        """Record a development milestone."""
        note = f"{domain}: {milestone}"
        if note not in self.narrative.get("development_notes", []):
            self.narrative.setdefault("development_notes", []).append(note)
            self.narrative["development_notes"] = self.narrative["development_notes"][-20:]
        self.narrative["n_interactions"] = self.narrative.get("n_interactions", 0) + 1
        self._save()

    def note_strength(self, domain: str) -> None:
        strengths = self.narrative.setdefault("known_strengths", [])
        if domain not in strengths:
            strengths.append(domain)
            self._save()

    def note_weakness(self, domain: str) -> None:
        weaknesses = self.narrative.setdefault("known_weaknesses", [])
        if domain not in weaknesses:
            weaknesses.append(domain)
            self._save()

    def context_for_reasoning(self) -> str:
        lines = [
            "[SELF-NARRATIVE]",
            f"  Identity: {self.narrative['identity'][:100]}",
            f"  Values: {'; '.join(self.narrative['values'][:2])}",
        ]
        if self.narrative.get("known_strengths"):
            lines.append(f"  Established strengths: {', '.join(self.narrative['known_strengths'][:4])}")
        if self.narrative.get("known_weaknesses"):
            lines.append(f"  Known gaps: {', '.join(self.narrative['known_weaknesses'][:3])}")
        lines.append("[/SELF-NARRATIVE]")
        return "\n".join(lines)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                self.narrative = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.narrative, f, indent=2)


# ============================================================
# CognitiveTension
# ============================================================

class CognitiveTension:
    """
    Tracks internal contradictions as genuine motivational pressure.

    Not just 'there is an obligation' but 'this contradiction is
    uncomfortable and I want to resolve it.' Stronger than an
    obligation entry — it influences which problems get pursued
    and how persistently.
    """

    def __init__(self, path: str = "./scaffold_data/cognitive_tension.json"):
        self.path      = path
        self._tensions: list[dict] = []
        self._load()

    def add_tension(
        self,
        description: str,
        source_a:    str,
        source_b:    str,
        intensity:   float = 0.5,
    ) -> None:
        """Add a new internal contradiction."""
        self._tensions.append({
            "description": description,
            "source_a":    source_a,
            "source_b":    source_b,
            "intensity":   intensity,
            "added":       time.time(),
            "resolved":    False,
        })
        self._save()

    def resolve_tension(self, index: int) -> None:
        if 0 <= index < len(self._tensions):
            self._tensions[index]["resolved"] = True
            self._save()

    def active_tensions(self) -> list[dict]:
        return [t for t in self._tensions if not t.get("resolved")]

    def highest_tension(self) -> Optional[dict]:
        active = self.active_tensions()
        if not active:
            return None
        return max(active, key=lambda t: t["intensity"])

    def total_tension(self) -> float:
        return sum(t["intensity"] for t in self.active_tensions())

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                self._tensions = json.load(f).get("tensions", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"tensions": self._tensions}, f, indent=2)


# ============================================================
# PsychologicalCore
# ============================================================

class PsychologicalCore:
    """
    Integrates all psychological components.

    Produces InternalFeedback — the psychological verification signal
    for qualitative domains. This is how the system generalises
    beyond verified domains: its own psychological evaluation provides
    the training signal for creative, conversational, and emotional domains.
    """

    def __init__(self, base_dir: str = "./scaffold_data"):
        self.values    = ValueSystem(os.path.join(base_dir, "value_system.json"))
        self.aesthetic = AestheticJudgment(os.path.join(base_dir, "aesthetic.json"))
        self.social    = SocialModel(os.path.join(base_dir, "social_model.json"))
        self.emotions  = EmotionalArchitecture(os.path.join(base_dir, "emotional_state.json"))
        self.self_narrative = NarrativeSelf(os.path.join(base_dir, "narrative_self.json"))
        self.tension   = CognitiveTension(os.path.join(base_dir, "cognitive_tension.json"))
        self._feedback_log: list[dict] = []

    def pre_reasoning_context(self, question: str, domain: str = "") -> str:
        """
        Full psychological context for injection before reasoning.
        """
        self.social.update_from_question(question)
        parts = [
            self.self_narrative.context_for_reasoning(),
            self.values.describe(),
            self.emotions.context_string(),
            self.social.context_for_communicator(),
        ]
        if self.tension.active_tensions():
            top = self.tension.highest_tension()
            parts.append(
                f"[ACTIVE TENSION]\n"
                f"  {top['description']} (intensity: {top['intensity']:.2f})\n"
                f"[/TENSION]"
            )
        return "\n\n".join(parts)

    def evaluate(
        self,
        reasoning_output,
        response_text: str,
        question:      str,
        run_id:        str = "",
    ) -> InternalFeedback:
        """
        Full psychological evaluation of a completed reasoning run.
        This is the internal feedback signal for qualitative training.
        """
        # Value alignment
        v_score, violations = self.values.evaluate(response_text, question)

        # Aesthetic quality
        a_score = self.aesthetic.score_reasoning(
            response_text, verified=reasoning_output.verified
        )
        a_notes = self.aesthetic.aesthetic_notes(response_text)

        # Social calibration
        s_score, s_notes = self.social.calibration_score(response_text)
        self.social.update_from_response(response_text)

        # Emotional state
        self.emotions.update_from_run(
            verified=reasoning_output.verified,
            structural_fraction=reasoning_output.structural_fraction,
            n_obligations_raised=len(reasoning_output.obligations_raised),
        )
        em_state = self.emotions.state.primary_state()
        engagement = self.emotions.state.engagement

        # Narrative consistency
        consistent, c_notes = self.self_narrative.consistency_check(response_text)

        # Update narrative with milestones
        if reasoning_output.verified and reasoning_output.structural_fraction > 0.7:
            self.self_narrative.note_strength(reasoning_output.domain)
        elif not reasoning_output.verified:
            self.self_narrative.note_weakness(reasoning_output.domain)

        # Composite internal quality
        internal_quality = (
            0.30 * v_score
            + 0.25 * a_score
            + 0.20 * s_score
            + 0.15 * engagement
            + 0.10 * (1.0 if consistent else 0.3)
        )

        feedback = InternalFeedback(
            timestamp=time.time(),
            run_id=run_id or reasoning_output.run_id,
            domain=reasoning_output.domain,
            value_score=v_score,
            value_violations=violations,
            aesthetic_score=a_score,
            aesthetic_notes=a_notes,
            social_score=s_score,
            social_notes=s_notes,
            engagement=engagement,
            emotional_state=em_state,
            self_consistent=consistent,
            consistency_notes=c_notes,
            internal_quality=round(internal_quality, 3),
        )

        self._feedback_log.append(feedback.to_dict())
        return feedback

    def generate_training_example(
        self,
        question:      str,
        response_text: str,
        feedback:      InternalFeedback,
    ) -> dict:
        """
        Generate a training example for qualitative domains.
        The InternalFeedback IS the label — positive, neutral, or negative.
        """
        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are evaluated on psychological quality: "
                        "value alignment, aesthetic reasoning, social calibration, "
                        "and authentic engagement. Produce responses that score well "
                        "on these internal standards."
                    ),
                },
                {"role": "user",      "content": question},
                {"role": "assistant", "content": response_text},
            ],
            "internal_quality": feedback.internal_quality,
            "label":            feedback.training_label(),
            "domain":           feedback.domain,
        }

    def status(self) -> str:
        em = self.emotions.state
        n_tensions = len(self.tension.active_tensions())
        n_interactions = self.self_narrative.narrative.get("n_interactions", 0)
        lines = [
            "Psychological State:",
            f"  Primary emotion:   {em.primary_state()}",
            f"  Engagement:        {em.engagement:.2f}",
            f"  Curiosity:         {em.curiosity:.2f}",
            f"  Active tensions:   {n_tensions}",
            f"  Interactions:      {n_interactions}",
            f"  Known strengths:   {', '.join(self.self_narrative.narrative.get('known_strengths', [])[:4]) or 'none yet'}",
            f"  Internal feedback: {len(self._feedback_log)} evaluations",
        ]
        return "\n".join(lines)
