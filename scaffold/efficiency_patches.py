"""
efficiency_patches.py
=====================

Three quick wins that don't require architectural changes:

  1. SQLite WAL mode — prevents write locks blocking reads under
     concurrent users. One PRAGMA, massive difference at >2 users.

  2. Groq model routing — use a faster model for simple questions,
     the full 70b only for complex reasoning.

  3. Response caching — identical questions return cached responses
     instantly. Uses an LRU cache keyed on (question_hash, domain).

Apply these by calling apply_all_patches() at startup in web_api_monetized.py.
"""

import hashlib
import time
from collections import OrderedDict
from typing import Optional


# ============================================================
# 1. SQLite WAL mode
# ============================================================

def apply_wal_mode(db_path: str = "./bri.db") -> None:
    """
    Enable Write-Ahead Logging on the SQLite database.
    
    Without WAL: each write locks the database, blocking all reads.
    With WAL: reads and writes proceed concurrently.
    
    Makes a noticeable difference at >2 simultaneous users.
    Safe to call multiple times (idempotent).
    """
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=10000;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.close()
        print(f"[efficiency] SQLite WAL mode enabled: {db_path}")
    except Exception as e:
        print(f"[efficiency] WAL mode warning: {e}")


# ============================================================
# 2. Groq model routing by complexity
# ============================================================

# Model speeds (approximate tokens/sec on Groq)
GROQ_MODELS = {
    "fast":    "llama-3.1-8b-instant",      # ~1200 tok/s — simple questions
    "balanced": "mixtral-8x7b-32768",        # ~600 tok/s  — most questions
    "full":    "llama-3.3-70b-versatile",    # ~300 tok/s  — complex reasoning
}

COMPLEX_SIGNALS = [
    "derive", "prove", "show that", "analyse", "analyze",
    "from first principles", "what are the implications",
    "reconcile", "relationship between", "why does",
    "think through", "deeply", "fundamentally",
]

SIMPLE_SIGNALS = [
    "what is", "define", "who is", "when did", "how many",
    "what does", "name", "list", "examples of",
]


def select_groq_model(question: str, user_tier: str = "pro") -> str:
    """
    Select the appropriate Groq model for this question and tier.
    
    Fast model for simple questions: ~4x faster than 70b.
    Balanced model for most Pro users.
    Full 70b only for complex API-tier reasoning.
    """
    lower = question.lower()
    words = len(lower.split())

    # Simple question: use fast model regardless of tier
    if words < 8 or any(s in lower for s in SIMPLE_SIGNALS):
        if not any(c in lower for c in COMPLEX_SIGNALS):
            return GROQ_MODELS["fast"]

    # Complex question: use full model for API tier, balanced for Pro
    if any(c in lower for c in COMPLEX_SIGNALS):
        if user_tier == "api":
            return GROQ_MODELS["full"]
        return GROQ_MODELS["balanced"]

    # Default: balanced for Pro, fast for Free (Free uses Ollama anyway)
    return GROQ_MODELS["balanced"]


# ============================================================
# 3. Response cache (LRU)
# ============================================================

class ResponseCache:
    """
    LRU cache for identical questions.
    
    Questions are hashed — the question text isn't stored.
    Cache hits return the previous response instantly.
    Cache is invalidated when the system learns significantly
    (belief revision, new crystallised knowledge).
    
    Max size: 200 entries. TTL: 1 hour.
    """

    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl      = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._hits    = 0
        self._misses  = 0

    def _key(self, question: str, domain: str) -> str:
        text = f"{question.lower().strip()}|{domain}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, question: str, domain: str) -> Optional[dict]:
        key = self._key(question, domain)
        if key not in self._cache:
            self._misses += 1
            return None
        entry = self._cache[key]
        if time.time() - entry["cached_at"] > self.ttl:
            del self._cache[key]
            self._misses += 1
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return entry["response"]

    def set(self, question: str, domain: str, response: dict) -> None:
        # Only cache verified responses
        if not response.get("verified"):
            return
        key = self._key(question, domain)
        self._cache[key] = {
            "response":  response,
            "cached_at": time.time(),
        }
        self._cache.move_to_end(key)
        # Evict oldest if over limit
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def invalidate_domain(self, domain: str) -> None:
        """Call when beliefs in a domain are revised."""
        # Can't easily invalidate by domain without storing it
        # Simple approach: clear the whole cache (it rebuilds quickly)
        self._cache.clear()

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size":      len(self._cache),
            "max_size":  self.max_size,
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(self._hits / max(1, total), 3),
        }


# ============================================================
# Apply all patches
# ============================================================

_response_cache: Optional[ResponseCache] = None

def apply_all_patches(db_path: str = "./bri.db") -> ResponseCache:
    """
    Call once at startup in web_api_monetized.py:
    
        from efficiency_patches import apply_all_patches
        cache = apply_all_patches()
    
    Returns the ResponseCache instance for use in request handlers.
    """
    global _response_cache

    apply_wal_mode(db_path)

    _response_cache = ResponseCache(max_size=200, ttl_seconds=3600)
    print("[efficiency] Response cache initialised (200 entries, 1hr TTL)")
    print(f"[efficiency] Groq model routing: fast={GROQ_MODELS['fast']}")

    return _response_cache


def get_cache() -> Optional[ResponseCache]:
    return _response_cache
