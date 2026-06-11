"""
collective_learning.py
======================

Manages the shared intelligence layer of the multi-user system.

When multiple users work with the same scaffold, their learning
compounds. This module defines what is shared between all users
and what remains private to each session.

SHARED (benefits everyone):
  - Technique library           grows from all domains explored by any user
  - Belief system               updates from collective evidence across all users
  - Procedural patterns         automatic patterns once enough users verify them
  - Small model training data   verified runs from all opted-in users
  - Cross-domain insights       DMN spontaneous connections from all sessions

PRIVATE (stays with each user):
  - Conversation history        their specific exchanges
  - Working memory              current session state
  - Social model                system calibrated to their expertise and style
  - Episodic memory             their specific question history (opt-in to share)

Contribution is opt-in. Users benefit from shared knowledge
regardless of whether they contribute their own runs.

The key innovation: verified reasoning from one user immediately
improves the system for all other users in the same domain.
One person working on biology at 9am means the system is
better at biology for everyone at 10am.
"""

import os
import json
import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# UserSession — one user's private state
# ============================================================

@dataclass
class UserSession:
    user_id:           str
    created_at:        float = field(default_factory=time.time)
    last_active:       float = field(default_factory=time.time)
    domain:            str   = "general"
    n_questions:       int   = 0
    n_verified:        int   = 0
    opted_in:          bool  = True     # contribute to shared pool
    conversation:      list  = field(default_factory=list)
    working_memory_path: str = ""
    social_model_path: str   = ""

    def to_dict(self) -> dict:
        return {
            "user_id":     self.user_id,
            "created_at":  self.created_at,
            "last_active": self.last_active,
            "domain":      self.domain,
            "n_questions": self.n_questions,
            "n_verified":  self.n_verified,
            "opted_in":    self.opted_in,
        }


# ============================================================
# SharedContribution — one verified run entered into shared pool
# ============================================================

@dataclass
class SharedContribution:
    run_id:             str
    user_id:            str          # anonymised
    domain:             str
    method:             str
    structural_fraction: float
    verified:           bool
    confidence:         str
    timestamp:          float
    # Content
    question_hash:      str          # hash of question (not stored in clear)
    training_example:   dict         # the actual training data

    def to_dict(self) -> dict:
        return {
            "run_id":             self.run_id,
            "user_id":            self.user_id,
            "domain":             self.domain,
            "method":             self.method,
            "structural_fraction": self.structural_fraction,
            "verified":           self.verified,
            "confidence":         self.confidence,
            "timestamp":          self.timestamp,
            "question_hash":      self.question_hash,
            "training_example":   self.training_example,
        }


# ============================================================
# CollectiveLearningHub
# ============================================================

class CollectiveLearningHub:
    """
    Manages shared intelligence across all users.

    Every verified run from an opted-in user goes into the shared
    pool. The pool feeds: the small model training, the technique
    library, the belief system, and the procedural memory.

    The shared components improve continuously from all users.
    The private components (working memory, social model) stay
    per-user.

    Thread-safe: multiple concurrent users supported.
    """

    def __init__(
        self,
        base_dir:  str  = "./shared_data",
        verbose:   bool = True,
    ):
        self.base_dir = base_dir
        self.verbose  = verbose
        self._lock    = threading.Lock()

        # Session management
        self._sessions:     dict[str, UserSession] = {}

        # Shared data paths
        self.shared_pool_path   = os.path.join(base_dir, "shared_pool.jsonl")
        self.sessions_path      = os.path.join(base_dir, "sessions.json")
        self.stats_path         = os.path.join(base_dir, "collective_stats.json")

        # Stats
        self._stats = {
            "total_contributions": 0,
            "total_users":         0,
            "domains_active":      {},
            "total_verified":      0,
            "phi_collective":      0.0,
        }

        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "users"), exist_ok=True)
        self._load_sessions()
        self._load_stats()

    # ── Session management ─────────────────────────────────

    def create_session(
        self,
        user_id:  Optional[str] = None,
        domain:   str           = "general",
        opted_in: bool          = True,
    ) -> UserSession:
        """Create a new user session."""
        if user_id is None:
            user_id = self._generate_id()

        user_dir = os.path.join(self.base_dir, "users", user_id)
        os.makedirs(user_dir, exist_ok=True)

        session = UserSession(
            user_id=user_id,
            domain=domain,
            opted_in=opted_in,
            working_memory_path=os.path.join(user_dir, "working_memory.json"),
            social_model_path=os.path.join(user_dir, "social_model.json"),
        )

        with self._lock:
            self._sessions[user_id] = session
            self._stats["total_users"] = len(self._sessions)

        self._save_sessions()

        if self.verbose:
            print(f"[collective] New session: {user_id[:8]}... "
                  f"(domain={domain}, opt_in={opted_in})")

        return session

    def get_session(self, user_id: str) -> Optional[UserSession]:
        return self._sessions.get(user_id)

    def update_session_activity(
        self, user_id: str, verified: bool = False
    ) -> None:
        session = self._sessions.get(user_id)
        if session:
            session.last_active = time.time()
            session.n_questions += 1
            if verified:
                session.n_verified += 1
            self._save_sessions()

    def active_sessions(self, within_minutes: int = 60) -> list[UserSession]:
        """Return sessions active in the last N minutes."""
        cutoff = time.time() - within_minutes * 60
        return [s for s in self._sessions.values()
                if s.last_active > cutoff]

    # ── Contribution ───────────────────────────────────────

    def contribute(
        self,
        reasoning_output,
        question:         str,
        user_id:          str,
        training_example: dict,
    ) -> bool:
        """
        Add a verified run to the shared pool.
        Only called when user has opted in.

        Returns True if contribution was accepted.
        """
        session = self._sessions.get(user_id)
        if not session or not session.opted_in:
            return False
        if not reasoning_output.verified:
            return False

        # Anonymise the question (store hash only, not text)
        q_hash = hashlib.sha256(question.encode()).hexdigest()[:16]

        contribution = SharedContribution(
            run_id=reasoning_output.run_id,
            user_id=self._anonymise(user_id),
            domain=reasoning_output.domain,
            method=reasoning_output.method,
            structural_fraction=reasoning_output.structural_fraction,
            verified=True,
            confidence=reasoning_output.confidence,
            timestamp=time.time(),
            question_hash=q_hash,
            training_example=training_example,
        )

        with self._lock:
            with open(self.shared_pool_path, "a") as f:
                f.write(json.dumps(contribution.to_dict()) + "\n")

            self._stats["total_contributions"] += 1
            self._stats["total_verified"]      += 1
            domain = reasoning_output.domain
            self._stats["domains_active"][domain] = (
                self._stats["domains_active"].get(domain, 0) + 1
            )

        self._save_stats()

        if self.verbose:
            total = self._stats["total_contributions"]
            print(f"  [collective] Contribution #{total}: "
                  f"{domain} — {reasoning_output.method[:40]}")

        return True

    def get_shared_pool(
        self,
        domain:     str   = "",
        limit:      int   = 100,
        since:      float = 0,
    ) -> list[dict]:
        """Read from the shared pool, optionally filtered."""
        if not os.path.exists(self.shared_pool_path):
            return []
        results = []
        try:
            with open(self.shared_pool_path) as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if domain and item.get("domain") != domain:
                            continue
                        if since and item.get("timestamp", 0) < since:
                            continue
                        results.append(item)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return results[-limit:]

    def shared_training_examples(self, limit: int = 500) -> list[dict]:
        """All training examples from the shared pool for the small model."""
        pool = self.get_shared_pool(limit=limit)
        return [item["training_example"] for item in pool
                if "training_example" in item]

    # ── Collective intelligence queries ────────────────────

    def domain_depth(self, domain: str) -> dict:
        """How much collective knowledge exists for a domain?"""
        pool = self.get_shared_pool(domain=domain)
        return {
            "n_contributions":   len(pool),
            "n_verified":        sum(1 for p in pool if p.get("verified")),
            "methods":           list({p.get("method","") for p in pool})[:5],
            "mean_structural":   (
                sum(p.get("structural_fraction", 0) for p in pool)
                / max(1, len(pool))
            ),
            "depth":             (
                "deep"     if len(pool) > 50 else
                "moderate" if len(pool) > 15 else
                "shallow"  if len(pool) > 3  else
                "none"
            ),
        }

    def collective_phi(self) -> float:
        """
        Proxy for collective Φ: how integrated is the shared knowledge?
        More users, more domains, more recent activity = higher collective Φ.
        """
        active    = len(self.active_sessions(within_minutes=60))
        domains   = len(self._stats.get("domains_active", {}))
        total     = self._stats.get("total_contributions", 0)

        phi = (
            min(1.0, active / 10.0) * 0.3         # active users
            + min(1.0, domains / 8.0) * 0.3       # domain breadth
            + min(1.0, total / 200.0) * 0.4       # contribution depth
        )
        self._stats["phi_collective"] = round(phi, 3)
        return round(phi, 3)

    def stats(self) -> dict:
        """Current collective learning statistics."""
        return {
            **self._stats,
            "active_sessions":  len(self.active_sessions()),
            "phi_collective":   self.collective_phi(),
            "domain_counts":    self._stats.get("domains_active", {}),
        }

    def format_shared_context(
        self, question: str, domain: str
    ) -> str:
        """
        Format shared collective knowledge relevant to this question
        as context for the Reasoner.
        """
        depth = self.domain_depth(domain)
        if depth["n_contributions"] == 0:
            return ""

        active = len(self.active_sessions())
        lines = [
            "[COLLECTIVE KNOWLEDGE]",
            f"  Domain depth ({domain}): {depth['depth']} "
            f"({depth['n_contributions']} contributions from "
            f"{self._stats['total_users']} users)",
        ]
        if depth["methods"]:
            lines.append(
                f"  Known methods: {', '.join(depth['methods'][:3])}"
            )
        if active > 1:
            lines.append(
                f"  Currently active: {active} user(s)"
            )
        lines.append(
            f"  Collective Φ: {self.collective_phi():.3f}"
        )
        lines.append("[/COLLECTIVE]")
        return "\n".join(lines)

    # ── Internals ─────────────────────────────────────────

    def _generate_id(self) -> str:
        import secrets
        return secrets.token_hex(12)

    def _anonymise(self, user_id: str) -> str:
        return hashlib.sha256(user_id.encode()).hexdigest()[:12]

    def _load_sessions(self) -> None:
        if not os.path.exists(self.sessions_path):
            return
        try:
            with open(self.sessions_path) as f:
                data = json.load(f)
            for d in data.get("sessions", []):
                uid  = d["user_id"]
                udir = os.path.join(self.base_dir, "users", uid)
                s    = UserSession(
                    user_id=uid,
                    created_at=d.get("created_at", time.time()),
                    last_active=d.get("last_active", time.time()),
                    domain=d.get("domain", "general"),
                    opted_in=d.get("opted_in", True),
                    n_questions=d.get("n_questions", 0),
                    n_verified=d.get("n_verified", 0),
                    working_memory_path=os.path.join(udir, "working_memory.json"),
                    social_model_path=os.path.join(udir, "social_model.json"),
                )
                self._sessions[uid] = s
        except (json.JSONDecodeError, OSError):
            pass

    def _save_sessions(self) -> None:
        with open(self.sessions_path, "w") as f:
            json.dump({
                "sessions": [s.to_dict() for s in self._sessions.values()]
            }, f, indent=2)

    def _load_stats(self) -> None:
        if not os.path.exists(self.stats_path):
            return
        try:
            with open(self.stats_path) as f:
                self._stats.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass

    def _save_stats(self) -> None:
        with open(self.stats_path, "w") as f:
            json.dump(self._stats, f, indent=2)
