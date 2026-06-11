"""
belief_system.py
================

The plasticity layer of the psychological architecture.

Every belief the system holds has a confidence level that updates
based on evidence. When failures accumulate around a specific domain
or problem type, the system actively questions the assumptions
underlying its approach — not just "which step went wrong" but
"is my entire way of thinking about this correct?"

Five components:

  BeliefSystem           — holds all beliefs with confidence levels.
                           Beliefs update bidirectionally: success
                           strengthens, failure weakens. When confidence
                           drops below threshold, the belief enters a
                           questioning state.

  AssumptionAuditor      — fires when repeated failures suggest the
                           framework is wrong rather than the execution.
                           Generates explicit questions about each
                           assumption underlying the current approach.
                           Proposes alternative framings.

  BeliefRevisionEngine   — the formal updating mechanism. Tracks
                           evidence for and against each belief. When
                           enough counter-evidence accumulates, initiates
                           belief revision. Logs every revision so the
                           temporal self can see how the system's thinking
                           has changed.

  PhilosophicalInquiry   — the deepest level of questioning. Not just
                           "which assumption is wrong" but "are my
                           foundational ways of approaching this class
                           of problem correct?" Can question ontological,
                           methodological, and epistemological assumptions
                           in any domain.

  EpistemicPlasticityMonitor — tracks how rigid or plastic the system
                           is being over time. Detects when it is stuck
                           in a belief that is no longer serving it.
                           Flags when the same assumptions have been
                           held for too long without being tested.

Together these make the psychology genuinely evolving rather than
rigidly fixed. The values, aesthetic judgments, and self-narrative
can all be questioned and revised when evidence warrants it.
"""

import os
import re
import json
import math
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# BeliefNode — one thing the system believes
# ============================================================

@dataclass
class BeliefNode:
    """
    One belief held with a confidence level.

    Confidence updates based on evidence. When it drops low enough
    the belief enters questioning_state — the system actively looks
    for alternatives rather than continuing to apply a belief it
    no longer trusts.
    """
    belief_id:          str
    statement:          str         # what is believed
    domain:             str         # what domain this applies to
    belief_type:        str         # "methodological"|"ontological"|"aesthetic"|"value"|"empirical"
    confidence:         float       # 0-1, starts at prior
    prior:              float       # starting confidence
    times_supported:    int = 0     # how many times this belief led to success
    times_failed:       int = 0     # how many times this belief led to failure
    consecutive_failures: int = 0   # resets on success
    last_tested:        float = field(default_factory=time.time)
    last_revised:       Optional[float] = None
    questioning_state:  bool = False
    source:             str = "assumed"  # "assumed"|"derived"|"revised"|"inherited"
    evidence_for:       list = field(default_factory=list)
    evidence_against:   list = field(default_factory=list)
    revision_history:   list = field(default_factory=list)

    QUESTIONING_THRESHOLD  = 0.35   # below this, belief is actively questioned
    REVISION_THRESHOLD     = 0.20   # below this, belief is revised or abandoned
    CONSECUTIVE_FAIL_LIMIT = 3      # consecutive failures trigger questioning

    def update_success(self, evidence: str = "") -> None:
        self.times_supported    += 1
        self.consecutive_failures = 0
        self.confidence          = min(1.0, self.confidence + 0.06)
        self.last_tested         = time.time()
        self.questioning_state   = False
        if evidence:
            self.evidence_for.append(evidence[:120])
            self.evidence_for = self.evidence_for[-10:]

    def update_failure(self, evidence: str = "") -> None:
        self.times_failed         += 1
        self.consecutive_failures  += 1
        # Bayesian-inspired decay: more failures → faster decay
        decay = 0.08 + min(0.12, self.consecutive_failures * 0.03)
        self.confidence = max(0.0, self.confidence - decay)
        self.last_tested = time.time()
        if evidence:
            self.evidence_against.append(evidence[:120])
            self.evidence_against = self.evidence_against[-10:]
        # Enter questioning state if threshold crossed
        if (self.confidence < self.QUESTIONING_THRESHOLD
                or self.consecutive_failures >= self.CONSECUTIVE_FAIL_LIMIT):
            self.questioning_state = True

    def revise(self, new_statement: str, reason: str) -> None:
        self.revision_history.append({
            "from":      self.statement[:100],
            "to":        new_statement[:100],
            "reason":    reason[:100],
            "timestamp": time.time(),
            "confidence_at_revision": self.confidence,
        })
        self.statement       = new_statement
        self.confidence      = self.prior * 0.7  # start fresh but not from zero
        self.times_failed    = 0
        self.times_supported = 0
        self.consecutive_failures = 0
        self.questioning_state   = False
        self.source              = "revised"
        self.last_revised        = time.time()

    @property
    def success_rate(self) -> float:
        total = self.times_supported + self.times_failed
        return self.times_supported / max(1, total)

    def to_dict(self) -> dict:
        return {
            "belief_id":           self.belief_id,
            "statement":           self.statement,
            "domain":              self.domain,
            "belief_type":         self.belief_type,
            "confidence":          round(self.confidence, 3),
            "prior":               self.prior,
            "times_supported":     self.times_supported,
            "times_failed":        self.times_failed,
            "consecutive_failures":self.consecutive_failures,
            "last_tested":         self.last_tested,
            "last_revised":        self.last_revised,
            "questioning_state":   self.questioning_state,
            "source":              self.source,
            "evidence_for":        self.evidence_for[-5:],
            "evidence_against":    self.evidence_against[-5:],
            "revision_history":    self.revision_history[-5:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BeliefNode":
        node = cls(
            belief_id=d["belief_id"],
            statement=d["statement"],
            domain=d.get("domain", "general"),
            belief_type=d.get("belief_type", "empirical"),
            confidence=d.get("confidence", 0.7),
            prior=d.get("prior", 0.7),
        )
        for k in ["times_supported","times_failed","consecutive_failures",
                  "last_tested","last_revised","questioning_state","source",
                  "evidence_for","evidence_against","revision_history"]:
            if k in d:
                setattr(node, k, d[k])
        return node


# ============================================================
# BeliefSystem
# ============================================================

# Seed beliefs — the system starts with these and they evolve
SEED_BELIEFS = [
    {
        "belief_id":   "b-method-ansatz",
        "statement":   "Ansatz-based reasoning — postulating a form and constraining it — is effective for finding closure relations",
        "domain":      "physics",
        "belief_type": "methodological",
        "confidence":  0.75,
        "prior":       0.75,
        "source":      "assumed",
    },
    {
        "belief_id":   "b-method-verification",
        "statement":   "Symbolic verification with sympy is a reliable test of whether a derivation is correct",
        "domain":      "general",
        "belief_type": "methodological",
        "confidence":  0.85,
        "prior":       0.85,
        "source":      "assumed",
    },
    {
        "belief_id":   "b-structure-transfer",
        "statement":   "Structural reasoning patterns transfer across domains — what works in physics can work in economics",
        "domain":      "general",
        "belief_type": "ontological",
        "confidence":  0.65,
        "prior":       0.65,
        "source":      "assumed",
    },
    {
        "belief_id":   "b-epistemic-labels",
        "statement":   "Labelling steps as [STRUCTURAL], [DERIVED], [VERIFIABLE] improves reasoning quality",
        "domain":      "general",
        "belief_type": "methodological",
        "confidence":  0.80,
        "prior":       0.80,
        "source":      "assumed",
    },
    {
        "belief_id":   "b-aesthetic-clarity",
        "statement":   "Clear, unambiguous reasoning is more valuable than evocative but imprecise reasoning",
        "domain":      "general",
        "belief_type": "aesthetic",
        "confidence":  0.70,
        "prior":       0.70,
        "source":      "assumed",
    },
    {
        "belief_id":   "b-newtonian-limit",
        "statement":   "Physical theories must recover Newtonian mechanics in the weak-field low-velocity limit",
        "domain":      "physics",
        "belief_type": "empirical",
        "confidence":  0.90,
        "prior":       0.90,
        "source":      "assumed",
    },
    {
        "belief_id":   "b-bio-emergence",
        "statement":   "Biological systems have emergent properties that resist step-by-step derivation",
        "domain":      "biology",
        "belief_type": "ontological",
        "confidence":  0.60,
        "prior":       0.60,
        "source":      "assumed",
    },
]


class BeliefSystem:
    """
    Holds all beliefs the system has formed, each with a confidence level.

    Beliefs update from every reasoning run. When confidence drops,
    the belief enters questioning state. When a belief is revised,
    the revision is logged for temporal self-awareness.
    """

    def __init__(self, path: str = "./scaffold_data/belief_system.json"):
        self.path    = path
        self._beliefs: dict[str, BeliefNode] = {}
        self._load()
        if not self._beliefs:
            self._seed()

    def get(self, belief_id: str) -> Optional[BeliefNode]:
        return self._beliefs.get(belief_id)

    def all_beliefs(self) -> list[BeliefNode]:
        return list(self._beliefs.values())

    def beliefs_for_domain(self, domain: str) -> list[BeliefNode]:
        return [b for b in self._beliefs.values()
                if b.domain in (domain, "general")]

    def questioning_beliefs(self) -> list[BeliefNode]:
        return [b for b in self._beliefs.values() if b.questioning_state]

    def weakest_beliefs(self, domain: str = "", n: int = 3) -> list[BeliefNode]:
        beliefs = self.beliefs_for_domain(domain) if domain else self.all_beliefs()
        return sorted(beliefs, key=lambda b: b.confidence)[:n]

    def update_from_outcome(
        self,
        domain:   str,
        verified: bool,
        method:   str = "",
        evidence: str = "",
    ) -> list[str]:
        """
        Update beliefs relevant to this domain/method outcome.
        Returns list of belief IDs that entered questioning state.
        """
        newly_questioning: list[str] = []
        relevant = self.beliefs_for_domain(domain)

        for belief in relevant:
            # Methodological beliefs: update if method matches
            if belief.belief_type == "methodological":
                if method and any(w in belief.statement.lower()
                                  for w in method.lower().split()[:3]):
                    if verified:
                        belief.update_success(evidence)
                    else:
                        belief.update_failure(evidence)
                        if belief.questioning_state:
                            newly_questioning.append(belief.belief_id)

            # Empirical beliefs: update from verification
            elif belief.belief_type == "empirical":
                if verified:
                    belief.update_success(evidence)
                else:
                    belief.update_failure(evidence)
                    if belief.questioning_state:
                        newly_questioning.append(belief.belief_id)

        self._save()
        return newly_questioning

    def add_belief(
        self,
        statement:   str,
        domain:      str,
        belief_type: str,
        confidence:  float = 0.6,
        source:      str   = "derived",
    ) -> str:
        bid = f"b-{domain[:6]}-{int(time.time() * 100) % 10000}"
        self._beliefs[bid] = BeliefNode(
            belief_id=bid,
            statement=statement,
            domain=domain,
            belief_type=belief_type,
            confidence=confidence,
            prior=confidence,
            source=source,
        )
        self._save()
        return bid

    def revise_belief(
        self, belief_id: str, new_statement: str, reason: str
    ) -> bool:
        if belief_id not in self._beliefs:
            return False
        self._beliefs[belief_id].revise(new_statement, reason)
        self._save()
        return True

    def _seed(self) -> None:
        for b in SEED_BELIEFS:
            self._beliefs[b["belief_id"]] = BeliefNode.from_dict(b)
        self._save()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for d in data.get("beliefs", []):
                node = BeliefNode.from_dict(d)
                self._beliefs[node.belief_id] = node
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(
                {"beliefs": [b.to_dict() for b in self._beliefs.values()]},
                f, indent=2,
            )


# ============================================================
# AssumptionAuditor
# ============================================================

# Domain-specific assumption generators
ASSUMPTION_QUESTION_TEMPLATES = {
    "physics": [
        "Am I assuming {method} is the right approach for this class of problem?",
        "Is the Newtonian limit actually applicable here, or am I assuming it without checking?",
        "Have I assumed that dimensional analysis will constrain this sufficiently?",
        "Am I assuming the symmetry of this system is higher than it actually is?",
        "Is the weak-field approximation genuinely valid in this regime?",
        "Am I treating this as equilibrium when it might be a genuinely non-equilibrium system?",
    ],
    "biology": [
        "Am I assuming this biological system can be modelled with a top-down approach?",
        "Have I assumed separability of subsystems that might actually be entangled?",
        "Am I looking for a single mechanism where there might be multiple competing ones?",
        "Is my assumption that this system is near equilibrium warranted?",
        "Am I imposing a physics-style derivation on a system that is fundamentally historical?",
    ],
    "economics": [
        "Am I assuming rational agents when the behaviour might be better explained otherwise?",
        "Have I assumed stationarity in what might be a non-stationary system?",
        "Is the representative agent framework actually valid here?",
        "Am I assuming this market mechanism is the relevant one?",
        "Have I assumed the equilibrium is unique when there might be multiple?",
    ],
    "general": [
        "What is the core assumption underlying my approach that I have not explicitly stated?",
        "If the opposite of my main assumption were true, what would that imply?",
        "Am I confusing correlation with causation in how I have framed this?",
        "What framework am I using without realising it — and is it the right one?",
        "Am I pattern-matching to something familiar when this might be genuinely novel?",
        "What would I need to believe for my current approach to be wrong?",
        "Have I assumed this problem is well-posed when it might not be?",
    ],
}


class AssumptionAuditor:
    """
    Fires when repeated failures suggest the framework is wrong.

    When a domain or problem type has accumulated enough failures,
    this generates explicit questions about the assumptions underlying
    the current approach. Not "what step went wrong" but "what are
    we taking for granted that might be false?"

    Also generates alternative framings — concrete alternative hypotheses
    about what the right approach might be.
    """

    FAILURE_THRESHOLD = 3    # consecutive failures before auditing

    def __init__(
        self,
        belief_system:   BeliefSystem,
        path:            str = "./scaffold_data/assumption_audit.json",
    ):
        self.beliefs    = belief_system
        self.path       = path
        self._audit_log: list[dict] = []
        self._load()

    def should_audit(self, domain: str, consecutive_failures: int) -> bool:
        """Should we audit assumptions for this domain right now?"""
        return consecutive_failures >= self.FAILURE_THRESHOLD

    def audit_assumptions(
        self,
        domain:           str,
        method:           str,
        question:         str,
        n_failures:       int,
    ) -> dict:
        """
        Full assumption audit for a domain/method that keeps failing.

        Returns:
          questions:     explicit questions about assumptions
          alternatives:  concrete alternative framings to try
          weak_beliefs:  beliefs that might be the problem
          audit_note:    summary
        """
        # Get domain-specific questions
        domain_templates = ASSUMPTION_QUESTION_TEMPLATES.get(
            domain, ASSUMPTION_QUESTION_TEMPLATES["general"]
        )
        general_templates = ASSUMPTION_QUESTION_TEMPLATES["general"]

        # Fill in method-specific placeholders
        questions = []
        for tmpl in (domain_templates + general_templates)[:6]:
            q = tmpl.replace("{method}", method or "this approach")
            questions.append(q)

        # Find weakest beliefs for this domain — these are the candidates
        weak = self.beliefs.weakest_beliefs(domain, n=3)
        weak_belief_statements = [b.statement for b in weak]

        # Generate alternatives based on what the failures suggest
        alternatives = self._generate_alternatives(domain, method, weak)

        # Log this audit
        audit = {
            "timestamp":    time.time(),
            "domain":       domain,
            "method":       method,
            "n_failures":   n_failures,
            "questions":    questions,
            "alternatives": alternatives,
            "weak_beliefs": weak_belief_statements,
        }
        self._audit_log.append(audit)
        self._save()

        audit_note = (
            f"After {n_failures} failures in {domain}: "
            f"auditing {len(questions)} assumptions, "
            f"proposing {len(alternatives)} alternatives. "
            f"Weakest belief: '{weak[0].statement[:60]}' "
            f"(confidence {weak[0].confidence:.2f})"
            if weak else f"After {n_failures} failures in {domain}: "
                         f"no strong candidate belief identified."
        )

        return {
            "questions":    questions,
            "alternatives": alternatives,
            "weak_beliefs": weak_belief_statements,
            "audit_note":   audit_note,
        }

    def _generate_alternatives(
        self,
        domain:  str,
        method:  str,
        weak:    list[BeliefNode],
    ) -> list[str]:
        """Generate concrete alternative framings to try."""
        alternatives = []

        # Alternative based on weakest belief
        if weak:
            b = weak[0]
            if "ansatz" in b.statement.lower():
                alternatives.append(
                    "Try a dimensional analysis approach instead of an ansatz — constrain from units rather than postulating a form"
                )
            elif "verification" in b.statement.lower():
                alternatives.append(
                    "Try a different verification method — numerical rather than symbolic, or check limiting cases rather than full derivation"
                )
            elif "transfer" in b.statement.lower() or "structure" in b.statement.lower():
                alternatives.append(
                    "Treat this domain as fundamentally different rather than structurally analogous to physics — look for domain-specific patterns"
                )
            elif "equilibrium" in b.statement.lower():
                alternatives.append(
                    "Model this as a non-equilibrium system — consider time-dependent or path-dependent approaches"
                )

        # Generic alternatives
        generic = [
            f"Approach {domain} problems bottom-up (from specific cases to general) rather than top-down (from assumed principles)",
            f"Try the simplest possible model first in {domain} before adding complexity",
            f"Look for what is conserved or invariant in this problem rather than what is derived",
            f"Treat this as an existence question (does a solution exist?) before asking what it is",
        ]
        alternatives.extend(random.sample(generic, min(2, len(generic))))

        return alternatives[:4]

    def questioning_context(self, domain: str, method: str = "") -> str:
        """
        Format the current questioning state as context for reasoning.
        Used when the system is in assumption-questioning mode.
        """
        weak  = self.beliefs.weakest_beliefs(domain, n=2)
        if not weak or weak[0].confidence > 0.5:
            return ""

        lines = [
            "[ASSUMPTION QUESTIONING MODE]",
            f"  My approach in {domain} has been struggling.",
            f"  Beliefs under question:",
        ]
        for b in weak:
            if b.confidence < 0.5:
                lines.append(
                    f"    - '{b.statement[:80]}' (confidence: {b.confidence:.2f}, "
                    f"failed {b.times_failed}×)"
                )
        lines.append(
            "  I should consider whether my assumptions are the problem, "
            "not my execution."
        )
        lines.append("[/QUESTIONING MODE]")
        return "\n".join(lines)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                self._audit_log = json.load(f).get("audits", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"audits": self._audit_log[-30:]}, f, indent=2)


# ============================================================
# BeliefRevisionEngine
# ============================================================

class BeliefRevisionEngine:
    """
    Formal mechanism for revising beliefs based on accumulated evidence.

    When enough counter-evidence has built up against a belief, this
    proposes a revision. The revision is logged. The old belief and
    the new one are both retained in history so the temporal self
    can track how thinking has changed.

    This is what makes the psychology genuinely evolving rather than
    accumulating data on top of fixed foundations.
    """

    def __init__(
        self,
        belief_system: BeliefSystem,
        path:          str = "./scaffold_data/belief_revisions.json",
    ):
        self.beliefs   = belief_system
        self.path      = path
        self._revisions: list[dict] = []
        self._load()

    def check_for_needed_revisions(self) -> list[dict]:
        """
        Scan all beliefs for ones that warrant revision.
        Returns list of revision proposals.
        """
        proposals = []
        for belief in self.beliefs.all_beliefs():
            if belief.confidence <= BeliefNode.REVISION_THRESHOLD:
                proposal = self._propose_revision(belief)
                if proposal:
                    proposals.append(proposal)
        return proposals

    def apply_revision(
        self,
        belief_id:     str,
        new_statement: str,
        reason:        str,
    ) -> bool:
        success = self.beliefs.revise_belief(belief_id, new_statement, reason)
        if success:
            self._revisions.append({
                "timestamp":      time.time(),
                "belief_id":      belief_id,
                "new_statement":  new_statement,
                "reason":         reason,
            })
            self._save()
        return success

    def revision_summary(self) -> str:
        if not self._revisions:
            return "No belief revisions have occurred yet."
        n = len(self._revisions)
        recent = self._revisions[-1]
        return (
            f"{n} belief revision(s) total. "
            f"Most recent: '{recent['new_statement'][:60]}' "
            f"({time.strftime('%Y-%m-%d', time.localtime(recent['timestamp']))})"
        )

    def _propose_revision(self, belief: BeliefNode) -> Optional[dict]:
        """Generate a revision proposal for a failing belief."""
        if not belief.evidence_against:
            return None
        return {
            "belief_id":          belief.belief_id,
            "current_statement":  belief.statement,
            "confidence":         belief.confidence,
            "times_failed":       belief.times_failed,
            "evidence_against":   belief.evidence_against[-2:],
            "suggested_revision": self._generate_revision(belief),
            "reason":             (
                f"Confidence dropped to {belief.confidence:.2f} after "
                f"{belief.times_failed} failures"
            ),
        }

    def _generate_revision(self, belief: BeliefNode) -> str:
        """Generate a revised version of a belief."""
        stmt    = belief.statement
        domain  = belief.domain
        btype   = belief.belief_type

        # Soften absolute claims
        for absolute in ["always", "is always", "will always", "must", "is reliable"]:
            if absolute in stmt.lower():
                return stmt.replace(absolute, f"often (but not always)")

        # Add domain qualification
        if domain != "general" and domain not in stmt.lower():
            return f"{stmt} — but this may not hold in all {domain} contexts"

        # Add epistemic qualification
        return f"{stmt} — though this belief should be held more tentatively given recent evidence"

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                self._revisions = json.load(f).get("revisions", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"revisions": self._revisions}, f, indent=2)


# ============================================================
# PhilosophicalInquiry
# ============================================================

PHILOSOPHICAL_QUESTIONS = {
    "ontological": [
        "Is the thing I am reasoning about actually the kind of thing I think it is?",
        "Have I assumed a level of description (micro/macro) that may not be the right one?",
        "Am I treating something as an object when it might be better described as a process?",
        "Is the boundary I have drawn around this problem the right boundary?",
        "Have I assumed independence between things that might be genuinely entangled?",
    ],
    "methodological": [
        "Is the method I am using appropriate for this kind of question?",
        "What would I have to believe about the world for this method to be the right one?",
        "Am I applying a method because it is correct here or because it is familiar?",
        "What does this method make invisible — what can it not see?",
        "If I used a completely different method, would I get a different answer, and which would be more true?",
    ],
    "epistemological": [
        "What would it mean for this claim to be true versus merely coherent?",
        "Am I confusing a framework's internal consistency with correspondence to reality?",
        "What evidence would cause me to abandon this approach entirely?",
        "Have I tested the foundations of this reasoning, or only the superstructure?",
        "Is this knowledge — or is it a useful fiction that has not yet been refuted?",
    ],
    "axiological": [
        "Am I valuing the right things in how I am approaching this?",
        "Is the standard by which I am judging success the correct standard for this problem?",
        "Would someone with different values approach this differently — and might they be right?",
        "What am I optimising for, and should I be optimising for something else?",
    ],
}


class PhilosophicalInquiry:
    """
    The deepest level of questioning.

    Not just 'which assumption is wrong' but 'is my entire way of
    approaching this class of problem correct?'

    This fires when:
    - A domain has been failing despite assumption auditing
    - A belief revision has occurred more than once in the same domain
    - The system's own model of a domain has been shown to be systematically wrong

    Generates questions at four levels: ontological (what is this?),
    methodological (am I using the right approach?), epistemological
    (how would I know?), and axiological (am I valuing the right things?).

    These questions are intended to be genuinely destabilising —
    to interrupt comfortable reasoning and force a deeper reconsideration.
    """

    def __init__(self, path: str = "./scaffold_data/philosophical_inquiry.json"):
        self.path          = path
        self._inquiries:   list[dict] = []
        self._insight_log: list[dict] = []
        self._load()

    def trigger_inquiry(
        self,
        domain:          str,
        trigger_reason:  str,
        failed_approach: str = "",
    ) -> dict:
        """
        Trigger a philosophical inquiry for a domain.
        Returns structured inquiry with questions at all four levels.
        """
        inquiry = {
            "timestamp":        time.time(),
            "domain":           domain,
            "trigger_reason":   trigger_reason,
            "failed_approach":  failed_approach,
            "questions": {
                level: self._select_questions(level, domain, failed_approach)
                for level in ["ontological", "methodological", "epistemological", "axiological"]
            },
            "core_challenge": self._generate_core_challenge(domain, failed_approach),
            "reframing_invitation": self._reframing_invitation(domain),
        }

        self._inquiries.append(inquiry)
        self._save()
        return inquiry

    def record_insight(
        self,
        domain:  str,
        insight: str,
        triggered_by: str = "",
    ) -> None:
        """Record an insight that emerged from philosophical inquiry."""
        self._insight_log.append({
            "timestamp":    time.time(),
            "domain":       domain,
            "insight":      insight[:200],
            "triggered_by": triggered_by[:100],
        })
        self._save()

    def format_for_reasoning(self, domain: str) -> str:
        """
        If the system is in philosophical inquiry mode for this domain,
        format the inquiry as context for the next reasoning attempt.
        """
        recent = [i for i in self._inquiries if i["domain"] == domain]
        if not recent:
            return ""

        latest  = recent[-1]
        q_list  = []
        for level, questions in latest["questions"].items():
            if questions:
                q_list.append(f"  [{level.upper()}] {questions[0]}")

        lines = [
            "[PHILOSOPHICAL INQUIRY ACTIVE]",
            f"  My usual approach in {domain} may need deeper examination.",
            f"  Core challenge: {latest['core_challenge'][:100]}",
            "  Live questions:",
        ] + q_list + [
            f"  Invitation: {latest['reframing_invitation'][:100]}",
            "[/PHILOSOPHICAL INQUIRY]",
        ]
        return "\n".join(lines)

    def insights_for(self, domain: str) -> list[str]:
        return [
            i["insight"] for i in self._insight_log
            if i["domain"] == domain
        ]

    def _select_questions(
        self, level: str, domain: str, failed_approach: str
    ) -> list[str]:
        pool = PHILOSOPHICAL_QUESTIONS.get(level, [])
        selected = []
        for q in pool:
            q_filled = q.replace("{domain}", domain).replace("{approach}", failed_approach)
            selected.append(q_filled)
        return selected[:2]

    def _generate_core_challenge(self, domain: str, failed_approach: str) -> str:
        challenges = [
            f"Is the category '{domain}' itself the right way to carve up this problem space?",
            f"Does '{failed_approach}' presuppose a structure that may not be there?",
            f"What is the most fundamental assumption in my '{domain}' reasoning that I have never explicitly tested?",
            f"If an expert from a completely different field looked at this, what would they find obviously wrong?",
        ]
        return challenges[len(self._inquiries) % len(challenges)]

    def _reframing_invitation(self, domain: str) -> str:
        invitations = [
            f"Try to describe the {domain} problem in a language that has no {domain}-specific terms.",
            "Imagine you had to explain why your approach failed to someone who had never heard of this field.",
            "What is the simplest possible model of this situation, and does it already contain the answer?",
            "What would you do if the approach you have been using were provably wrong?",
        ]
        return invitations[len(self._inquiries) % len(invitations)]

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._inquiries   = d.get("inquiries", [])
            self._insight_log = d.get("insights", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "inquiries": self._inquiries[-20:],
                "insights":  self._insight_log[-50:],
            }, f, indent=2)


# ============================================================
# EpistemicPlasticityMonitor
# ============================================================

class EpistemicPlasticityMonitor:
    """
    Tracks how rigid or plastic the system is being over time.

    A system that never revises its beliefs is dogmatic.
    A system that revises them too easily is unstable.
    The goal is calibrated plasticity: beliefs held firmly when
    supported, revised readily when evidence warrants it, with
    increasing willingness to revise as failures accumulate.

    Reports:
    - Current plasticity score (0=rigid, 1=appropriately plastic)
    - Which domains are most rigid
    - Which beliefs have been held longest without testing
    - Whether the system is trending toward rigidity
    """

    def __init__(
        self,
        belief_system:  BeliefSystem,
        revision_engine: BeliefRevisionEngine,
        path:           str = "./scaffold_data/plasticity_monitor.json",
    ):
        self.beliefs  = belief_system
        self.revision = revision_engine
        self.path     = path
        self._log:    list[dict] = []
        self._load()

    def compute_plasticity(self) -> dict:
        """
        Compute current epistemic plasticity across all domains.
        """
        beliefs = self.beliefs.all_beliefs()
        if not beliefs:
            return {"score": 0.5, "status": "no beliefs yet"}

        now = time.time()

        # How many beliefs are in questioning state
        questioning_rate = sum(1 for b in beliefs if b.questioning_state) / len(beliefs)

        # Average time since beliefs were last tested
        ages = [(now - b.last_tested) / 86400 for b in beliefs]
        mean_age_days = sum(ages) / len(ages)

        # How many revisions have occurred
        n_revisions  = len(self.revision._revisions)
        revision_rate = min(1.0, n_revisions / max(1, len(beliefs)) * 2)

        # Beliefs with very high confidence that have never failed
        untested_rigid = sum(
            1 for b in beliefs
            if b.confidence > 0.85 and b.times_failed == 0 and b.times_supported < 3
        )
        rigidity_penalty = untested_rigid / max(1, len(beliefs))

        # Plasticity score
        plasticity = (
            0.30 * questioning_rate
            + 0.25 * min(1.0, revision_rate)
            + 0.25 * min(1.0, 1.0 - rigidity_penalty)
            + 0.20 * min(1.0, mean_age_days / 7.0)   # older untested beliefs = more rigid
        )
        plasticity = round(max(0.0, min(1.0, plasticity)), 3)

        if plasticity < 0.2:
            status = "RIGID — beliefs not being updated despite use"
        elif plasticity < 0.4:
            status = "SOMEWHAT RIGID — limited belief revision occurring"
        elif plasticity < 0.7:
            status = "APPROPRIATELY PLASTIC — beliefs evolving with evidence"
        else:
            status = "HIGHLY PLASTIC — beliefs updating frequently"

        record = {
            "timestamp":        now,
            "plasticity":       plasticity,
            "status":           status,
            "questioning_rate": round(questioning_rate, 3),
            "n_revisions":      n_revisions,
            "untested_rigid":   untested_rigid,
            "mean_age_days":    round(mean_age_days, 1),
        }
        self._log.append(record)
        self._save()
        return record

    def rigidity_alert(self) -> Optional[str]:
        """Return an alert if the system is becoming too rigid."""
        if not self._log:
            return None
        recent = self._log[-5:]
        if len(recent) < 3:
            return None
        trend = [r["plasticity"] for r in recent]
        if all(trend[i] >= trend[i+1] for i in range(len(trend)-1)):
            return (
                f"[RIGIDITY ALERT] Epistemic plasticity has been declining: "
                f"{trend[0]:.2f} → {trend[-1]:.2f}. "
                f"Some beliefs may need active testing."
            )
        return None

    def most_rigid_domain(self) -> Optional[str]:
        """Which domain has the most untested high-confidence beliefs?"""
        domain_rigidity: dict[str, float] = {}
        for b in self.beliefs.all_beliefs():
            d = b.domain
            if b.confidence > 0.75 and b.times_failed == 0:
                domain_rigidity[d] = domain_rigidity.get(d, 0) + 1
        if not domain_rigidity:
            return None
        return max(domain_rigidity, key=domain_rigidity.__getitem__)

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


# ============================================================
# PlasticPsychologicalCore — integrates belief system with psychology
# ============================================================

class PlasticPsychologicalCore:
    """
    Integrates the belief system with the full psychological architecture.

    This is the version of the psychological core that is genuinely
    plastic — beliefs evolve, assumptions are questioned when failures
    accumulate, philosophical inquiry fires when domains are persistently
    difficult, and the system's self-narrative updates when its beliefs
    about its own capabilities are revised.

    Usage alongside FullPsychologicalBundle:

        from belief_system import PlasticPsychologicalCore
        from psychology_extended import FullPsychologicalBundle

        psych   = FullPsychologicalBundle(base_dir="./scaffold_data")
        plastic = PlasticPsychologicalCore(base_dir="./scaffold_data")

        # Before reasoning — includes belief/questioning context
        ctx = plastic.pre_reasoning_context(question, domain)

        # After run — updates beliefs, may trigger inquiry
        events = plastic.process_run(reasoning_output, response, question)

        # Check if philosophical inquiry is active
        if events.get("philosophical_inquiry"):
            print("System is questioning its foundations in", domain)
    """

    # How many consecutive domain failures before philosophical inquiry
    PHILOSOPHICAL_TRIGGER = 6

    def __init__(self, base_dir: str = "./scaffold_data"):
        os.makedirs(base_dir, exist_ok=True)
        p = lambda n: os.path.join(base_dir, n)

        self.beliefs    = BeliefSystem(p("belief_system.json"))
        self.auditor    = AssumptionAuditor(self.beliefs, p("assumption_audit.json"))
        self.revision   = BeliefRevisionEngine(self.beliefs, p("belief_revisions.json"))
        self.philosophy = PhilosophicalInquiry(p("philosophical_inquiry.json"))
        self.plasticity = EpistemicPlasticityMonitor(
            self.beliefs, self.revision, p("plasticity_monitor.json")
        )
        self._domain_failures: dict[str, int] = {}

    def pre_reasoning_context(self, question: str, domain: str = "") -> str:
        """
        Full belief-aware context for injection before reasoning.
        Includes any active questioning or philosophical inquiry.
        """
        parts = []

        # Assumption questioning mode
        if domain and self._domain_failures.get(domain, 0) >= AssumptionAuditor.FAILURE_THRESHOLD:
            parts.append(self.auditor.questioning_context(domain))

        # Philosophical inquiry mode
        if domain and self._domain_failures.get(domain, 0) >= self.PHILOSOPHICAL_TRIGGER:
            parts.append(self.philosophy.format_for_reasoning(domain))

        # Beliefs currently being questioned
        questioning = self.beliefs.questioning_beliefs()
        if questioning:
            lines = ["[BELIEFS UNDER QUESTION]"]
            for b in questioning[:3]:
                lines.append(
                    f"  '{b.statement[:80]}' "
                    f"(confidence {b.confidence:.2f}, failed {b.times_failed}×)"
                )
            lines.append("[/UNDER QUESTION]")
            parts.append("\n".join(lines))

        # Rigidity alert
        alert = self.plasticity.rigidity_alert()
        if alert:
            parts.append(alert)

        return "\n\n".join(p for p in parts if p)

    def process_run(
        self,
        reasoning_output,
        response_text: str,
        question:      str,
    ) -> dict:
        """
        Process a completed run through the belief system.
        Updates beliefs, triggers audits, may initiate philosophical inquiry.

        Returns dict of events that occurred.
        """
        domain   = reasoning_output.domain
        verified = reasoning_output.verified
        method   = reasoning_output.method

        events: dict = {"domain": domain, "verified": verified}

        # Update domain failure tracking
        if verified:
            self._domain_failures[domain] = max(
                0, self._domain_failures.get(domain, 0) - 1
            )
        else:
            self._domain_failures[domain] = \
                self._domain_failures.get(domain, 0) + 1

        n_consec_failures = self._domain_failures.get(domain, 0)

        # Update beliefs
        newly_questioning = self.beliefs.update_from_outcome(
            domain=domain,
            verified=verified,
            method=method,
            evidence=question[:80],
        )
        events["beliefs_newly_questioning"] = newly_questioning

        # Assumption audit if threshold reached
        if not verified and self.auditor.should_audit(domain, n_consec_failures):
            audit = self.auditor.audit_assumptions(
                domain=domain,
                method=method,
                question=question,
                n_failures=n_consec_failures,
            )
            events["assumption_audit"] = audit
            print(
                f"\n  [belief_system] Assumption audit triggered: "
                f"{audit['audit_note']}"
            )
            print("  Questions:")
            for q in audit["questions"][:3]:
                print(f"    - {q}")
            print("  Alternatives:")
            for a in audit["alternatives"][:2]:
                print(f"    → {a}")

        # Philosophical inquiry if deeper threshold reached
        if n_consec_failures >= self.PHILOSOPHICAL_TRIGGER:
            inquiry = self.philosophy.trigger_inquiry(
                domain=domain,
                trigger_reason=f"{n_consec_failures} consecutive failures despite assumption auditing",
                failed_approach=method,
            )
            events["philosophical_inquiry"] = inquiry
            print(
                f"\n  [belief_system] PHILOSOPHICAL INQUIRY triggered for {domain}:\n"
                f"  Core challenge: {inquiry['core_challenge'][:100]}"
            )

        # Check for needed belief revisions
        proposals = self.revision.check_for_needed_revisions()
        if proposals:
            events["revision_proposals"] = proposals
            for p in proposals[:2]:
                print(
                    f"  [belief_system] Revision proposed: "
                    f"'{p['current_statement'][:60]}' → "
                    f"'{p['suggested_revision'][:60]}'"
                )
                # Auto-apply revision
                self.revision.apply_revision(
                    p["belief_id"],
                    p["suggested_revision"],
                    p["reason"],
                )

        # Update plasticity monitor
        plasticity = self.plasticity.compute_plasticity()
        events["plasticity"] = plasticity["plasticity"]

        return events

    def status(self) -> str:
        plasticity = self.plasticity.compute_plasticity()
        questioning = self.beliefs.questioning_beliefs()
        revised    = len(self.revision._revisions)
        inquiries  = len(self.philosophy._inquiries)

        lines = [
            "Belief System Status:",
            f"  Total beliefs:      {len(self.beliefs.all_beliefs())}",
            f"  Under question:     {len(questioning)}",
            f"  Revised:            {revised}",
            f"  Inquiries:          {inquiries}",
            f"  Plasticity:         {plasticity['plasticity']:.3f} — {plasticity['status']}",
        ]
        most_rigid = self.plasticity.most_rigid_domain()
        if most_rigid:
            lines.append(f"  Most rigid domain:  {most_rigid}")
        if questioning:
            lines.append("  Currently questioning:")
            for b in questioning[:2]:
                lines.append(f"    - '{b.statement[:70]}' ({b.confidence:.2f})")
        return "\n".join(lines)
