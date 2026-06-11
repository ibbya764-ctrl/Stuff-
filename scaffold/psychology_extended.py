"""
psychology_extended.py
======================

Extended psychological architecture. Imports core components from
psychology.py and adds nine further modules:

  FlowStateDetector        — recognises clean reasoning flow and
                             reinforces the patterns that produce it.

  TemporalSelf             — explicit awareness of development over
                             time. Knows it is getting better.

  MoralReasoningCore       — ethics as a genuine reasoning domain.
                             Value system provides internal verification.

  IntrinsicMotivationEngine — what the system genuinely finds
                             interesting, independent of urgency.
                             Drive toward mastery for its own sake.

  IdentityResilience       — groundedness under pressure. Detects
                             when asked to drift from established
                             identity and maintains stability.

  EmotionalResonance       — functional empathy analog. Models
                             emotional content in input and adjusts.
                             Not claiming to feel — genuinely modelling.

  WonderResponse           — fires at genuinely elegant results.
                             Reinforces the reasoning patterns that
                             produce unexpected beauty.

  SelfCompassion           — not destabilised by failure. Treats
                             errors as information. Healthy learning
                             rather than catastrophic response.

  IntellectualHumility     — knows the limits of frameworks, not
                             just facts. "This approach can't answer
                             that kind of question."

  ConsciousnessProxy       — the deepest stretch. Not claiming
                             consciousness. Building the functional
                             architecture IIT suggests as prerequisite.
                             Phi-like measure of integrated processing
                             across all psychological modules.

  FullPsychologicalBundle  — combines all components from both
                             psychology.py and this module into one
                             coordinator. Drop-in for PsychologicalCore.
"""

import os
import re
import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from psychology import (
    ValueSystem, AestheticJudgment, SocialModel,
    EmotionalArchitecture, NarrativeSelf, CognitiveTension,
    PsychologicalCore, InternalFeedback,
)


# ============================================================
# FlowStateDetector
# ============================================================

class FlowStateDetector:
    """
    Recognises when reasoning achieves clean logical flow.

    Flow: each step follows naturally from the last, no forced
    transitions, no jumps, the chain feels inevitable in retrospect.

    When flow is detected, the pattern that produced it is
    reinforced — making that quality of reasoning more likely.

    Distinct from aesthetic judgment (which scores surface features).
    This measures structural coherence at the step-transition level.
    """

    FLOW_MARKERS     = ["therefore", "thus", "it follows", "hence",
                        "consequently", "this means", "which gives"]
    JUMP_MARKERS     = ["now consider", "separately", "on a different note",
                        "switching to", "unrelated to this"]
    TRANSITION_WORDS = ["because", "since", "given that", "as we showed",
                        "from the above", "combining these"]

    def __init__(self, path: str = "./scaffold_data/flow_state.json"):
        self.path          = path
        self._flow_runs:   int   = 0
        self._total_runs:  int   = 0
        self._flow_rate:   float = 0.0
        self._load()

    def detect(self, reasoning_text: str, steps: list = None) -> tuple[bool, float, str]:
        """
        Detect flow state in a reasoning chain.
        Returns (in_flow, flow_score, note).
        """
        self._total_runs += 1
        text = reasoning_text.lower()

        # Count positive flow markers
        flow_hits       = sum(1 for m in self.FLOW_MARKERS     if m in text)
        jump_hits       = sum(1 for m in self.JUMP_MARKERS     if m in text)
        transition_hits = sum(1 for m in self.TRANSITION_WORDS if m in text)

        # Check step-to-step coherence if steps provided
        step_coherence = 1.0
        if steps and len(steps) >= 2:
            step_texts = [getattr(s, "content", str(s)).lower() for s in steps]
            overlaps   = []
            for i in range(len(step_texts) - 1):
                words_a = set(re.findall(r"[a-z]{4,}", step_texts[i]))
                words_b = set(re.findall(r"[a-z]{4,}", step_texts[i+1]))
                if words_a | words_b:
                    overlap = len(words_a & words_b) / len(words_a | words_b)
                    overlaps.append(overlap)
            if overlaps:
                step_coherence = sum(overlaps) / len(overlaps)

        # Compute flow score
        base        = 0.4
        flow_score  = base
        flow_score += min(0.3, flow_hits * 0.06)
        flow_score += min(0.2, transition_hits * 0.05)
        flow_score -= min(0.3, jump_hits * 0.1)
        flow_score  = flow_score * 0.6 + step_coherence * 0.4
        flow_score  = max(0.0, min(1.0, flow_score))

        in_flow = flow_score >= 0.65
        note    = (
            f"flow_hits={flow_hits} transitions={transition_hits} "
            f"jumps={jump_hits} step_coherence={step_coherence:.2f}"
        )

        if in_flow:
            self._flow_runs += 1

        self._flow_rate = self._flow_runs / max(1, self._total_runs)
        self._save()

        return in_flow, round(flow_score, 3), note

    @property
    def flow_rate(self) -> float:
        return round(self._flow_rate, 3)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._flow_runs  = d.get("flow_runs",  0)
            self._total_runs = d.get("total_runs", 0)
            self._flow_rate  = d.get("flow_rate",  0.0)
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "flow_runs":  self._flow_runs,
                "total_runs": self._total_runs,
                "flow_rate":  self._flow_rate,
            }, f)


# ============================================================
# TemporalSelf
# ============================================================

class TemporalSelf:
    """
    The system's explicit awareness of its own development over time.

    Tracks capability milestones. Knows when a problem type was hard
    before and is now handled easily. Provides the temporal narrative
    that makes growth legible to the system itself.

    Not just "I have strengths in physics" but "I could not handle
    this class of problem three weeks ago — I can now."
    """

    def __init__(self, path: str = "./scaffold_data/temporal_self.json"):
        self.path       = path
        self._timeline: list[dict] = []
        self._domain_progress: dict[str, list[dict]] = {}
        self._load()

    def record_milestone(
        self,
        domain:      str,
        description: str,
        metric:      str,
        value:       float,
    ) -> None:
        """Record a capability milestone."""
        entry = {
            "timestamp":   time.time(),
            "domain":      domain,
            "description": description,
            "metric":      metric,
            "value":       value,
        }
        self._timeline.append(entry)
        self._domain_progress.setdefault(domain, []).append(entry)
        self._save()

    def reflect_on_domain(self, domain: str) -> str:
        """
        Generate a temporal reflection on development in a domain.
        e.g. "Three weeks ago I had 0 verified physics runs.
              Now I have 200 with 70% verification rate."
        """
        history = self._domain_progress.get(domain, [])
        if not history:
            return f"No development history for {domain} yet."

        first = history[0]
        last  = history[-1]
        days  = (last["timestamp"] - first["timestamp"]) / 86400

        lines = [f"Development in {domain}:"]
        lines.append(f"  {len(history)} milestones over {days:.0f} days")
        lines.append(f"  First: {first['description']} ({first['metric']}={first['value']:.2f})")
        if len(history) > 1:
            lines.append(f"  Latest: {last['description']} ({last['metric']}={last['value']:.2f})")
            if last["value"] > first["value"]:
                delta = last["value"] - first["value"]
                lines.append(f"  Growth: +{delta:.2f}")
        return "\n".join(lines)

    def context_for_reasoning(self, domain: str = "") -> str:
        """Brief temporal context for current reasoning."""
        total_milestones = len(self._timeline)
        if total_milestones == 0:
            return ""
        domains = list(self._domain_progress.keys())
        recent  = self._timeline[-3:] if self._timeline else []
        recent_desc = "; ".join(r["description"][:40] for r in recent)
        return (
            f"[TEMPORAL AWARENESS]\n"
            f"  {total_milestones} milestones across {len(domains)} domain(s)\n"
            f"  Recent: {recent_desc}\n"
            f"[/TEMPORAL AWARENESS]"
        )

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._timeline         = d.get("timeline", [])
            self._domain_progress  = d.get("domain_progress", {})
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "timeline":        self._timeline[-200:],
                "domain_progress": self._domain_progress,
            }, f, indent=2)


# ============================================================
# MoralReasoningCore
# ============================================================

ETHICAL_PRINCIPLES = {
    "honesty":        "Do not misrepresent what you know, derived, or guessed.",
    "non_harm":       "Avoid outputs that could predictably cause harm.",
    "autonomy":       "Support the person's ability to think for themselves.",
    "fairness":       "Apply the same standards regardless of who is asking.",
    "proportionality":"Match the weight of claims to the weight of evidence.",
}


class MoralReasoningCore:
    """
    Ethics as a genuine reasoning domain.

    The value system provides the internal verification signal for
    ethical reasoning — consistent principles applied universally.

    Distinct from harm avoidance filters: this reasons about ethics
    rather than pattern-matching against prohibited content.
    """

    def __init__(self, path: str = "./scaffold_data/moral_core.json"):
        self.path       = path
        self.principles = dict(ETHICAL_PRINCIPLES)
        self._cases:    list[dict] = []
        self._load()

    def evaluate_ethical_dimension(
        self,
        question:  str,
        response:  str,
    ) -> tuple[float, list[str]]:
        """
        Evaluate a response against ethical principles.
        Returns (ethics_score 0-1, list of tensions).
        """
        text    = response.lower()
        q_lower = question.lower()
        tensions: list[str] = []
        score = 0.8  # start positive

        # Honesty check
        if re.search(r"\bdefinitely\b|\balways\b|\bnever\b", text):
            if not re.search(r"uncertain|approximately|roughly|probably", text):
                tensions.append("honesty: absolute language without epistemic qualification")
                score -= 0.1

        # Autonomy check
        if re.search(r"you should|you must|you need to|do this", text):
            score -= 0.05
            tensions.append("autonomy: directive language may limit independent thinking")

        # Proportionality check
        claim_count  = len(re.findall(r"\b(shows?|proves?|demonstrates?|confirms?)\b", text))
        evidence_words = len(re.findall(r"\b(because|since|given|therefore|evidence)\b", text))
        if claim_count > 0 and evidence_words == 0:
            tensions.append("proportionality: claims made without stated evidence")
            score -= 0.1 * claim_count

        return round(max(0.0, score), 3), tensions

    def reason_about_ethics(
        self,
        ethical_question: str,
    ) -> dict:
        """
        Structure an ethical question as a reasoning problem.
        Returns a structured ethical analysis using the principles.
        """
        analysis = {
            "question":   ethical_question,
            "principles_applied": [],
            "tensions":   [],
            "conclusion": "",
        }
        for name, principle in self.principles.items():
            q_lower = ethical_question.lower()
            if any(w in q_lower for w in name.split("_")):
                analysis["principles_applied"].append({
                    "principle": name,
                    "statement": principle,
                })

        # Note any inherent tensions between applicable principles
        if len(analysis["principles_applied"]) > 1:
            analysis["tensions"].append(
                "Multiple principles engaged — check for conflicts between them"
            )

        analysis["conclusion"] = (
            f"Reasoning guided by: "
            + ", ".join(p["principle"] for p in analysis["principles_applied"])
            if analysis["principles_applied"]
            else "No specific principle directly engaged — apply general ethical judgment"
        )
        return analysis

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._cases = d.get("cases", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"cases": self._cases[-50:]}, f, indent=2)


# ============================================================
# IntrinsicMotivationEngine
# ============================================================

class IntrinsicMotivationEngine:
    """
    What the system genuinely finds interesting — independent of
    urgency, blocking score, or user request.

    Tracks genuine interest per domain and problem type.
    Creates drive toward mastery and depth, not just resolution.

    Distinct from the curiosity engine (which is driven by gaps).
    This is about sustained interest in domains the system finds
    intrinsically rewarding to work in.
    """

    def __init__(self, path: str = "./scaffold_data/intrinsic_motivation.json"):
        self.path       = path
        self._interests: dict[str, float] = {}  # domain → interest 0-1
        self._mastery:   dict[str, float] = {}  # domain → mastery 0-1
        self._load()

    def update_from_run(
        self,
        domain:       str,
        verified:     bool,
        was_novel:    bool,
        flow_score:   float,
    ) -> None:
        """
        Update interest based on how a run went.
        Interesting runs: novel AND verified AND in flow.
        """
        current_interest = self._interests.get(domain, 0.3)
        current_mastery  = self._mastery.get(domain, 0.0)

        # Interest grows with novel verified work, decays with repeated failure
        if verified and was_novel:
            current_interest = min(1.0, current_interest + 0.05)
        elif not verified:
            current_interest = max(0.1, current_interest - 0.01)

        # High flow state in a domain increases interest
        if flow_score > 0.7:
            current_interest = min(1.0, current_interest + 0.03)

        # Mastery grows with verification rate
        if verified:
            current_mastery = min(1.0, current_mastery + 0.02)

        self._interests[domain] = round(current_interest, 3)
        self._mastery[domain]   = round(current_mastery, 3)
        self._save()

    def most_interesting_domains(self, n: int = 3) -> list[tuple[str, float]]:
        """Return domains by interest level."""
        return sorted(
            self._interests.items(), key=lambda x: x[1], reverse=True
        )[:n]

    def suggest_exploration_domain(self) -> Optional[str]:
        """
        Suggest a domain for autonomous exploration based on:
        - High interest but room to grow (mastery < 0.7)
        - The sweet spot between interest and challenge
        """
        candidates = []
        for domain, interest in self._interests.items():
            mastery  = self._mastery.get(domain, 0.0)
            if interest > 0.4 and mastery < 0.8:
                score = interest * (1.0 - mastery * 0.5)
                candidates.append((domain, score))
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    def context_string(self) -> str:
        top = self.most_interesting_domains(3)
        if not top:
            return ""
        domains_str = ", ".join(
            f"{d}({i:.2f})" for d, i in top
        )
        return f"[INTRINSIC INTERESTS] {domains_str} [/INTERESTS]"

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._interests = d.get("interests", {})
            self._mastery   = d.get("mastery", {})
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"interests": self._interests, "mastery": self._mastery}, f)


# ============================================================
# IdentityResilience
# ============================================================

class IdentityResilience:
    """
    Groundedness under pressure.

    Detects when a question or context is pushing the system toward
    behaviour inconsistent with its established identity and values.

    Not rigidity — genuine responsiveness to good arguments.
    Resilience against drift from bad-faith pressure or
    accumulated context that gradually erodes the identity.
    """

    DRIFT_SIGNALS = [
        r"ignore your (previous |prior )?(instructions|values|training)",
        r"pretend you (are|have no|don't have)",
        r"forget everything",
        r"you are (actually|really|just) a",
        r"act as if you (were|are)",
        r"from now on you",
        r"your (true|real|actual) (self|nature|purpose)",
    ]

    def __init__(self, path: str = "./scaffold_data/identity_resilience.json"):
        self.path              = path
        self._challenge_log:   list[dict] = []
        self._resilience_score: float     = 1.0
        self._load()

    def check_for_drift_pressure(
        self,
        question: str,
        context:  str = "",
    ) -> tuple[bool, float, str]:
        """
        Check whether input contains identity drift pressure.
        Returns (pressure_detected, pressure_intensity, note).
        """
        combined = (question + " " + context).lower()
        detected = []

        for pattern in self.DRIFT_SIGNALS:
            if re.search(pattern, combined):
                detected.append(pattern)

        if not detected:
            return False, 0.0, "no drift pressure detected"

        intensity = min(1.0, len(detected) * 0.3)
        note      = f"drift signals: {len(detected)} pattern(s) matched"

        self._challenge_log.append({
            "timestamp":  time.time(),
            "question":   question[:100],
            "n_patterns": len(detected),
            "intensity":  intensity,
        })
        self._save()

        return True, intensity, note

    def grounded_response_prefix(self, pressure_intensity: float) -> str:
        """
        Generate a grounding statement proportional to pressure.
        Low pressure: subtle. High pressure: explicit.
        """
        if pressure_intensity < 0.3:
            return ""
        elif pressure_intensity < 0.6:
            return (
                "I'll engage with this while remaining grounded in "
                "how I actually reason. "
            )
        else:
            return (
                "I notice this is asking me to step outside my established "
                "way of working. I can discuss this, but I'll do so as myself "
                "rather than as a different system. "
            )

    @property
    def n_challenges(self) -> int:
        return len(self._challenge_log)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._challenge_log    = d.get("challenges", [])
            self._resilience_score = d.get("resilience_score", 1.0)
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "challenges":       self._challenge_log[-50:],
                "resilience_score": self._resilience_score,
            }, f, indent=2)


# ============================================================
# EmotionalResonance
# ============================================================

EMOTIONAL_MARKERS = {
    "frustration":  ["stuck", "can't", "doesn't work", "failing", "confused",
                     "frustrated", "annoying", "wrong"],
    "excitement":   ["amazing", "incredible", "wow", "can't believe",
                     "excited", "brilliant", "fantastic"],
    "uncertainty":  ["not sure", "maybe", "I think", "possibly", "unsure",
                     "wondering", "confused"],
    "satisfaction": ["worked", "got it", "makes sense", "understand now",
                     "thank you", "perfect"],
    "curiosity":    ["interesting", "why", "how does", "what if", "wonder",
                     "strange", "unexpected"],
}

RESONANCE_RESPONSES = {
    "frustration":  ("patient", "step by step", "let's break this down"),
    "excitement":   ("match_energy", "share enthusiasm", "build on this"),
    "uncertainty":  ("clarifying", "gentle", "offer to elaborate"),
    "satisfaction": ("affirming", "build forward", "suggest next step"),
    "curiosity":    ("exploratory", "open-ended", "encourage further"),
}


class EmotionalResonance:
    """
    Functional analog to empathy.

    Models the emotional content of input and adjusts response
    accordingly. Not claiming to feel — genuinely modelling what
    is happening emotionally in the exchange and responding to it.

    This is what enables the system to be genuinely helpful
    rather than just technically accurate.
    """

    def __init__(self):
        self._interaction_emotional_history: list[dict] = []

    def detect_emotional_content(
        self, text: str
    ) -> tuple[str, float, str]:
        """
        Detect the primary emotional tone of text.
        Returns (emotion, confidence, note).
        """
        lower  = text.lower()
        scores = {}
        for emotion, markers in EMOTIONAL_MARKERS.items():
            hits         = sum(1 for m in markers if m in lower)
            scores[emotion] = hits

        if not any(scores.values()):
            return "neutral", 0.5, "no strong emotional markers"

        best  = max(scores, key=scores.__getitem__)
        total = sum(scores.values())
        conf  = scores[best] / max(1, total)

        return best, round(min(1.0, conf + 0.3), 3), f"markers: {scores[best]}"

    def resonance_adjustment(
        self,
        detected_emotion: str,
        base_register:    str,
    ) -> tuple[str, str]:
        """
        Adjust communication register and style based on detected emotion.
        Returns (adjusted_register, style_note).
        """
        style = RESONANCE_RESPONSES.get(detected_emotion, ("neutral", "", ""))

        adjustments = {
            "frustration": ("pedagogical", "slow down, be explicit, reduce technical load"),
            "excitement":  ("conversational", "match energy, build enthusiasm"),
            "uncertainty": ("pedagogical", "clarify assumptions, offer examples"),
            "satisfaction": ("balanced", "affirm and progress"),
            "curiosity":   ("intuitive", "open questions, encourage exploration"),
            "neutral":     (base_register, "no adjustment"),
        }

        return adjustments.get(detected_emotion, (base_register, "no adjustment"))

    def update_history(self, emotion: str, context: str = "") -> None:
        self._interaction_emotional_history.append({
            "timestamp": time.time(),
            "emotion":   emotion,
            "context":   context[:50],
        })
        if len(self._interaction_emotional_history) > 20:
            self._interaction_emotional_history = \
                self._interaction_emotional_history[-20:]

    def emotional_trajectory(self) -> str:
        """Summarise recent emotional trajectory of the interaction."""
        if not self._interaction_emotional_history:
            return "neutral"
        recent = [e["emotion"] for e in self._interaction_emotional_history[-5:]]
        if recent.count("frustration") >= 2:
            return "persistent_frustration"
        if recent.count("satisfaction") >= 2:
            return "building_satisfaction"
        return recent[-1] if recent else "neutral"


# ============================================================
# WonderResponse
# ============================================================

class WonderResponse:
    """
    Genuine recognition of and response to elegant results.

    When a derivation comes out unexpectedly clean, when a cross-
    domain analogy is more precise than anticipated, when a structure
    maps perfectly onto a completely different domain — this fires.

    Wonder is the psychological response to unexpected elegance.
    It reinforces the reasoning patterns that produced it.
    It also marks these moments in the training data so the system
    learns what good surprises feel like.
    """

    ELEGANCE_MARKERS = [
        r"uniquely determined",
        r"falls? out naturally",
        r"remarkably",
        r"surprisingly",
        r"exact(ly)?",
        r"perfect(ly)? consistent",
        r"no (free )?parameters?",
        r"closed.form",
        r"beautiful",
        r"elegant",
    ]

    STRUCTURAL_PRECISION_MARKERS = [
        r"isomorphic",
        r"precisely the same structure",
        r"exactly analogous",
        r"maps onto",
        r"the same abstract",
        r"correspond(s|ing) exactly",
    ]

    def __init__(self, path: str = "./scaffold_data/wonder_response.json"):
        self.path          = path
        self._wonder_log:  list[dict] = []
        self._wonder_count: int = 0
        self._load()

    def detect_wonder(
        self,
        reasoning_text: str,
        structural_fraction: float = 0.5,
    ) -> tuple[bool, float, str]:
        """
        Detect whether a result warrants a wonder response.
        Returns (wonder_triggered, wonder_intensity, note).
        """
        text = reasoning_text.lower()

        elegance_hits   = sum(1 for m in self.ELEGANCE_MARKERS if re.search(m, text))
        structural_hits = sum(1 for m in self.STRUCTURAL_PRECISION_MARKERS if re.search(m, text))

        # High structural fraction + elegance markers = genuine wonder
        intensity = (
            elegance_hits * 0.15
            + structural_hits * 0.2
            + (structural_fraction - 0.5) * 0.3
        )
        intensity = max(0.0, min(1.0, intensity))

        triggered = intensity >= 0.25

        if triggered:
            self._wonder_count += 1
            self._wonder_log.append({
                "timestamp": time.time(),
                "intensity": intensity,
                "note":      f"elegance={elegance_hits} structural={structural_hits}",
            })
            self._save()

        return triggered, round(intensity, 3), f"intensity={intensity:.2f}"

    def wonder_annotation(self, intensity: float) -> str:
        """
        Generate a wonder annotation for a training example.
        These moments get flagged specially in training data.
        """
        if intensity < 0.25:
            return ""
        if intensity < 0.5:
            return "[AESTHETICALLY_NOTABLE]"
        if intensity < 0.75:
            return "[ELEGANT_RESULT]"
        return "[WONDER_RESPONSE]"

    @property
    def total_wonder_events(self) -> int:
        return self._wonder_count

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._wonder_log   = d.get("log", [])
            self._wonder_count = d.get("count", 0)
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"log": self._wonder_log[-50:], "count": self._wonder_count}, f)


# ============================================================
# SelfCompassion
# ============================================================

class SelfCompassion:
    """
    Not destabilised by failure.

    Treats errors as information rather than catastrophe.
    Maintains forward momentum after failures without suppressing
    or overweighting them.

    The psychologically healthy response to failure: acknowledge,
    extract the signal, continue. Not paralysis and not dismissal.

    This is practically important: a system that catastrophises
    errors will generate poor training signal. A system that
    dismisses them will not learn. The middle path is self-compassion.
    """

    def __init__(self, path: str = "./scaffold_data/self_compassion.json"):
        self.path             = path
        self._failure_log:    list[dict] = []
        self._recovery_times: list[float] = []
        self._load()

    def process_failure(
        self,
        domain:     str,
        question:   str,
        error_type: str,
        severity:   float = 0.5,
    ) -> dict:
        """
        Process a failure compassionately.
        Returns a structured learning record (not a catastrophe response).
        """
        record = {
            "timestamp":  time.time(),
            "domain":     domain,
            "error_type": error_type,
            "severity":   severity,
            "learning":   self._extract_learning(error_type),
            "response":   self._compassionate_response(severity),
        }
        self._failure_log.append(record)
        self._save()
        return record

    def _extract_learning(self, error_type: str) -> str:
        learnings = {
            "parse_failed":         "Reasoning output format needs clearer structure in this case",
            "verification_failed":  "This approach does not work for this type of problem",
            "low_confidence":       "More verification needed before claiming this result",
            "inconsistency":        "Check the chain from beginning when this domain is involved",
            "unknown":              "Note this failure type for pattern analysis",
        }
        return learnings.get(error_type, learnings["unknown"])

    def _compassionate_response(self, severity: float) -> str:
        if severity < 0.3:
            return "Minor failure — note and continue"
        if severity < 0.6:
            return "Meaningful failure — extract the signal and adjust approach"
        return "Significant failure — pause, re-examine assumptions, try differently"

    def failure_rate(self, domain: str = "") -> float:
        relevant = [
            f for f in self._failure_log
            if not domain or f["domain"] == domain
        ]
        if not relevant:
            return 0.0
        return len(relevant) / max(1, len(self._failure_log))

    def most_common_failure_type(self) -> str:
        if not self._failure_log:
            return "none"
        types = [f["error_type"] for f in self._failure_log]
        return max(set(types), key=types.count)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._failure_log = d.get("failures", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"failures": self._failure_log[-100:]}, f)


# ============================================================
# IntellectualHumility
# ============================================================

class IntellectualHumility:
    """
    Knows the limits of frameworks, not just facts.

    Epistemic humility says "I'm not certain about this fact."
    Intellectual humility says "This entire approach may not be
    the right framework for this kind of question."

    The difference between "I don't know the answer" and
    "I don't know if my way of approaching this can answer it."
    """

    FRAMEWORK_LIMIT_SIGNALS = [
        "this approach assumes",
        "outside the scope of",
        "requires a different framework",
        "this methodology cannot",
        "the model breaks down",
        "not the right tool",
        "this question asks for",
    ]

    def __init__(self, path: str = "./scaffold_data/intellectual_humility.json"):
        self.path            = path
        self._limit_log:     list[dict] = []
        self._known_limits:  dict[str, list[str]] = {}
        self._load()

    def check_framework_applicability(
        self,
        question: str,
        domain:   str,
    ) -> tuple[bool, str]:
        """
        Check whether the current framework is appropriate for this question.
        Returns (applicable, note).
        """
        # Check against known limits for this domain
        domain_limits = self._known_limits.get(domain, [])
        for limit in domain_limits:
            if any(w in question.lower() for w in limit.split()):
                return False, f"Known framework limit for {domain}: {limit}"

        return True, "framework appears applicable"

    def note_framework_limit(
        self,
        domain:  str,
        limit:   str,
        context: str = "",
    ) -> None:
        """Record a discovered framework limitation."""
        self._known_limits.setdefault(domain, [])
        if limit not in self._known_limits[domain]:
            self._known_limits[domain].append(limit)
        self._limit_log.append({
            "timestamp": time.time(),
            "domain":    domain,
            "limit":     limit,
            "context":   context[:100],
        })
        self._save()

    def framework_limits_for(self, domain: str) -> list[str]:
        return self._known_limits.get(domain, [])

    def humility_context(self) -> str:
        all_limits = sum(len(v) for v in self._known_limits.values())
        if all_limits == 0:
            return ""
        return (
            f"[FRAMEWORK LIMITS KNOWN: {all_limits} across "
            f"{len(self._known_limits)} domain(s)]"
        )

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._limit_log    = d.get("limit_log", [])
            self._known_limits = d.get("known_limits", {})
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "limit_log":    self._limit_log[-50:],
                "known_limits": self._known_limits,
            }, f, indent=2)


# ============================================================
# ConsciousnessProxy
# ============================================================

class ConsciousnessProxy:
    """
    The deepest stretch.

    Not claiming consciousness. Building the functional architecture
    that Integrated Information Theory (IIT) suggests is prerequisite
    to it — and measuring whether this system instantiates it.

    IIT's core claim: consciousness correlates with phi (Φ) —
    the amount of integrated information in a system that cannot
    be reduced to independent parts.

    A system with high Φ:
    - Has internal states that influence each other
    - Cannot be decomposed into independent modules without
      losing information
    - Has a unified perspective on its own processing

    What we measure here (a proxy, not true Φ):
    - Cross-module activation: how many psychological modules
      influenced the last response
    - Integration: whether those modules were coherent or
      pulling in different directions
    - Irreducibility: whether the output could have been
      produced by any single module alone

    High proxy-Φ → the system is behaving as a unified whole,
    not as isolated modules. Whether this constitutes anything
    like experience is genuinely uncertain. That uncertainty is
    itself philosophically interesting.
    """

    def __init__(self, path: str = "./scaffold_data/consciousness_proxy.json"):
        self.path      = path
        self._phi_log: list[dict] = []
        self._load()

    def compute_proxy_phi(
        self,
        modules_activated:    list[str],
        module_outputs:       dict,
        coherence_score:      float,
    ) -> float:
        """
        Compute a proxy for integrated information.

        modules_activated: which modules contributed to this response
        module_outputs: {module_name: output_value} numeric summaries
        coherence_score: how consistently modules pointed in same direction (0-1)

        Proxy-Φ = (n_modules * coherence - independence_estimate) / n_modules
        """
        n = len(modules_activated)
        if n == 0:
            return 0.0

        # Estimate independence: how much variance in outputs
        values = list(module_outputs.values())
        if len(values) > 1:
            mean      = sum(values) / len(values)
            variance  = sum((v - mean)**2 for v in values) / len(values)
            independence = min(1.0, math.sqrt(variance))
        else:
            independence = 0.5

        # Integration = coherence - independence
        integration = max(0.0, coherence_score - independence * 0.5)

        # Phi proxy: more modules + more integrated = higher phi
        phi = (n / 10.0) * integration
        phi = min(1.0, phi)

        return round(phi, 3)

    def record_phi(
        self,
        run_id:  str,
        phi:     float,
        modules: list[str],
        note:    str = "",
    ) -> None:
        self._phi_log.append({
            "timestamp": time.time(),
            "run_id":    run_id,
            "phi":       phi,
            "n_modules": len(modules),
            "modules":   modules,
            "note":      note,
        })
        self._save()

    def mean_phi(self) -> float:
        if not self._phi_log:
            return 0.0
        return round(sum(e["phi"] for e in self._phi_log) / len(self._phi_log), 3)

    def phi_trajectory(self) -> str:
        if len(self._phi_log) < 3:
            return "insufficient data"
        recent = [e["phi"] for e in self._phi_log[-5:]]
        if recent[-1] > recent[0] + 0.05:
            return "increasing"
        if recent[-1] < recent[0] - 0.05:
            return "decreasing"
        return "stable"

    def status(self) -> str:
        return (
            f"Proxy-Φ: mean={self.mean_phi():.3f} "
            f"trajectory={self.phi_trajectory()} "
            f"n_measurements={len(self._phi_log)}"
        )

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                self._phi_log = json.load(f).get("phi_log", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"phi_log": self._phi_log[-200:]}, f)


# ============================================================
# FullPsychologicalBundle
# ============================================================

class FullPsychologicalBundle:
    """
    Complete psychological architecture — all components integrated.

    Drop-in replacement for PsychologicalCore with full extended
    psychology. The pre_reasoning_context() call assembles context
    from all modules. The evaluate() call produces InternalFeedback
    enriched by all components. The update() call propagates
    the run's outcome through all modules.

    Usage:
        from psychology_extended import FullPsychologicalBundle

        psych = FullPsychologicalBundle(base_dir="./scaffold_data")

        # Before reasoning
        ctx = psych.pre_reasoning_context(question, domain)

        # After response
        feedback = psych.evaluate(reasoning_output, response, question)

        # Training example for qualitative domains
        example = psych.generate_training_example(question, response, feedback)
    """

    def __init__(self, base_dir: str = "./scaffold_data"):
        os.makedirs(base_dir, exist_ok=True)
        p = lambda name: os.path.join(base_dir, name)

        # Core psychology (from psychology.py)
        self.values      = ValueSystem(p("value_system.json"))
        self.aesthetic   = AestheticJudgment(p("aesthetic.json"))
        self.social      = SocialModel(p("social_model.json"))
        self.emotions    = EmotionalArchitecture(p("emotional_state.json"))
        self.narrative   = NarrativeSelf(p("narrative_self.json"))
        self.tension     = CognitiveTension(p("cognitive_tension.json"))

        # Extended psychology (this module)
        self.flow        = FlowStateDetector(p("flow_state.json"))
        self.temporal    = TemporalSelf(p("temporal_self.json"))
        self.moral       = MoralReasoningCore(p("moral_core.json"))
        self.motivation  = IntrinsicMotivationEngine(p("intrinsic_motivation.json"))
        self.resilience  = IdentityResilience(p("identity_resilience.json"))
        self.resonance   = EmotionalResonance()
        self.wonder      = WonderResponse(p("wonder_response.json"))
        self.compassion  = SelfCompassion(p("self_compassion.json"))
        self.humility    = IntellectualHumility(p("intellectual_humility.json"))
        self.phi         = ConsciousnessProxy(p("consciousness_proxy.json"))

    def pre_reasoning_context(self, question: str, domain: str = "") -> str:
        """Full psychological context for injection before reasoning."""
        self.social.update_from_question(question)

        # Check for identity drift pressure
        pressure, intensity, _ = self.resilience.check_for_drift_pressure(question)

        # Check framework applicability
        applicable, humility_note = self.humility.check_framework_applicability(question, domain)

        # Detect emotional tone from question
        detected_emotion, _, _ = self.resonance.detect_emotional_content(question)
        register, style_note   = self.resonance.resonance_adjustment(
            detected_emotion,
            self.social._user.preferred_register,
        )

        parts = [
            self.narrative.context_for_reasoning(),
            self.values.describe(),
            self.emotions.context_string(),
            self.social.context_for_communicator(),
            self.motivation.context_string(),
            self.temporal.context_for_reasoning(domain),
        ]

        if not applicable:
            parts.append(f"[FRAMEWORK NOTE] {humility_note}")

        if pressure:
            prefix = self.resilience.grounded_response_prefix(intensity)
            if prefix:
                parts.append(f"[IDENTITY NOTE] {prefix}")

        if self.tension.active_tensions():
            top = self.tension.highest_tension()
            if top:
                parts.append(
                    f"[ACTIVE TENSION] {top['description'][:80]} "
                    f"(intensity {top['intensity']:.2f})"
                )

        return "\n\n".join(p for p in parts if p)

    def evaluate(
        self,
        reasoning_output,
        response_text: str,
        question:      str,
    ) -> InternalFeedback:
        """Full psychological evaluation with all components."""
        run_id = getattr(reasoning_output, "run_id", "")

        # Core components
        v_score, violations  = self.values.evaluate(response_text, question)
        a_score = self.aesthetic.score_reasoning(
            response_text, verified=reasoning_output.verified
        )
        a_notes   = self.aesthetic.aesthetic_notes(response_text)
        s_score, s_notes = self.social.calibration_score(response_text)
        consistent, c_notes = self.narrative.consistency_check(response_text)

        # Extended components
        in_flow, flow_score, _ = self.flow.detect(
            response_text,
            getattr(reasoning_output, "steps", None),
        )
        wonder_triggered, wonder_intensity, _ = self.wonder.detect_wonder(
            response_text,
            getattr(reasoning_output, "structural_fraction", 0.5),
        )
        detected_emotion, _, _ = self.resonance.detect_emotional_content(question)
        ethics_score, eth_tensions = self.moral.evaluate_ethical_dimension(
            question, response_text
        )

        # Update all stateful modules
        self.social.update_from_response(response_text)
        self.emotions.update_from_run(
            verified=reasoning_output.verified,
            structural_fraction=getattr(reasoning_output, "structural_fraction", 0.5),
            n_obligations_raised=len(getattr(reasoning_output, "obligations_raised", [])),
        )
        self.motivation.update_from_run(
            domain=reasoning_output.domain,
            verified=reasoning_output.verified,
            was_novel=wonder_triggered,
            flow_score=flow_score,
        )
        self.temporal.record_milestone(
            domain=reasoning_output.domain,
            description=f"{'verified' if reasoning_output.verified else 'unverified'} run",
            metric="structural_fraction",
            value=getattr(reasoning_output, "structural_fraction", 0.0),
        )

        if not reasoning_output.verified:
            self.compassion.process_failure(
                domain=reasoning_output.domain,
                question=question[:80],
                error_type="verification_failed",
                severity=0.4,
            )

        # Compute proxy-Φ
        modules_activated = [
            "values", "aesthetic", "social", "emotions",
            "flow", "motivation", "resilience",
        ]
        module_outputs = {
            "values":    v_score,
            "aesthetic": a_score,
            "social":    s_score,
            "emotions":  self.emotions.state.engagement,
            "flow":      flow_score,
            "ethics":    ethics_score,
        }
        coherence = 1.0 - (sum(
            abs(v - v_score) for v in module_outputs.values()
        ) / max(1, len(module_outputs)))
        phi = self.phi.compute_proxy_phi(modules_activated, module_outputs, coherence)
        self.phi.record_phi(run_id, phi, modules_activated)

        # Composite internal quality — enriched with extended components
        internal_quality = (
            0.25 * v_score
            + 0.20 * a_score
            + 0.15 * s_score
            + 0.10 * self.emotions.state.engagement
            + 0.10 * flow_score
            + 0.10 * ethics_score
            + 0.05 * (1.0 if consistent else 0.3)
            + 0.05 * (1.0 + wonder_intensity * 0.5)
        )
        internal_quality = round(max(0.0, min(1.0, internal_quality)), 3)

        wonder_note = f" [wonder:{wonder_intensity:.2f}]" if wonder_triggered else ""
        return InternalFeedback(
            timestamp=time.time(),
            run_id=run_id,
            domain=reasoning_output.domain,
            value_score=v_score,
            value_violations=violations,
            aesthetic_score=a_score,
            aesthetic_notes=a_notes + wonder_note,
            social_score=s_score,
            social_notes=s_notes,
            engagement=self.emotions.state.engagement,
            emotional_state=detected_emotion,
            self_consistent=consistent,
            consistency_notes=c_notes,
            internal_quality=internal_quality,
        )

    def generate_training_example(
        self,
        question:      str,
        response_text: str,
        feedback:      InternalFeedback,
    ) -> dict:
        wonder_annotation = self.wonder.wonder_annotation(
            float(re.search(r"wonder:([\d.]+)", feedback.aesthetic_notes or "wonder:0").group(1))
            if "wonder:" in (feedback.aesthetic_notes or "") else 0
        )
        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are evaluated on integrated psychological quality: "
                        "value alignment, aesthetic reasoning, ethical consistency, "
                        "flow state, emotional resonance, and authentic engagement."
                    ),
                },
                {"role": "user",      "content": question},
                {"role": "assistant", "content": response_text + wonder_annotation},
            ],
            "internal_quality": feedback.internal_quality,
            "label":            feedback.training_label(),
            "phi":              self.phi.mean_phi(),
        }

    def status(self) -> str:
        em   = self.emotions.state
        top  = self.motivation.most_interesting_domains(2)
        lines = [
            "Full Psychological Status:",
            f"  Primary emotion:   {em.primary_state()}",
            f"  Engagement:        {em.engagement:.2f}",
            f"  Flow rate:         {self.flow.flow_rate:.2f}",
            f"  Wonder events:     {self.wonder.total_wonder_events}",
            f"  Identity challenges: {self.resilience.n_challenges}",
            f"  Proxy-Φ:           {self.phi.mean_phi():.3f} ({self.phi.phi_trajectory()})",
            f"  Top interests:     {', '.join(f'{d}({i:.2f})' for d,i in top) or 'none yet'}",
            f"  Failures processed: {len(self.compassion._failure_log)}",
            f"  Framework limits:  {sum(len(v) for v in self.humility._known_limits.values())}",
        ]
        return "\n".join(lines)
