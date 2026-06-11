"""
knowledge_seeker.py
===================

Connects the scaffold's internal reasoning to external knowledge.

Two capabilities:

  NoveltyDetector      — when the system derives something, checks whether
                         it is genuinely new or already known. Queries the
                         internal stores first (fast, free), then searches
                         external sources if the internal check is
                         inconclusive. Returns a typed NoveltyResult with
                         status (known / similar / novel) and what was found.

  VerificationBootstrapper — when the system encounters a new domain, uses
                         external search to find how verification works in
                         that domain, generates Python verification functions
                         using the CodeGenerator, tests them in the sandbox,
                         and adds them to the available tools. The system
                         builds its own verification infrastructure.

These two together close the loop between internal reasoning and the
broader body of human knowledge. Every new finding gets checked against
what already exists. Every new domain gets its own verification tools
derived from real methodology.

The search interface is pluggable — inject any function that takes a
query string and returns a list of {title, snippet, url} dicts.
The existing web_search_tool.py from the pipeline can be used directly.
"""

import re
import json
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# Novelty detection
# ============================================================

@dataclass
class NoveltyResult:
    """
    The result of checking whether a finding is genuinely new.
    """
    status:          str     # "known" | "similar" | "novel" | "inconclusive"
    confidence:      float   # 0-1
    internal_matches: list   # matches found in obligation store / episodic
    external_matches: list   # matches found in web search
    contradictions:  list    # results that contradict the finding
    recommendation:  str     # what to do next

    def is_novel(self) -> bool:
        return self.status in ("novel", "inconclusive")

    def summary(self) -> str:
        lines = [
            f"Novelty: {self.status.upper()} (confidence {self.confidence:.0%})"
        ]
        if self.internal_matches:
            lines.append(f"  Internal matches: {len(self.internal_matches)}")
        if self.external_matches:
            lines.append(f"  External matches: {len(self.external_matches)}")
        if self.contradictions:
            lines.append(f"  ⚠ Contradictions found: {len(self.contradictions)}")
        lines.append(f"  Recommendation: {self.recommendation}")
        return "\n".join(lines)


class NoveltyDetector:
    """
    Checks whether a finding is genuinely new.

    Internal check (fast, no search):
      - Episodic store: has the system solved something like this before?
      - Obligation store: is this a known gap or known result?
      - Technique library: does this match an existing technique?

    External check (uses search_fn):
      - Web search for the finding's key claim
      - Looks for supporting literature, contradicting results, prior art
    """

    def __init__(
        self,
        episodic_store,
        obligation_store,
        technique_library,
        search_fn:        Optional[Callable] = None,
        similarity_threshold: float = 0.35,
    ):
        self.episodic   = episodic_store
        self.store      = obligation_store
        self.library    = technique_library
        self.search_fn  = search_fn
        self.sim_thresh = similarity_threshold

    def check(
        self,
        claim:      str,
        domain:     str,
        context:    str = "",
        search:     bool = True,
    ) -> NoveltyResult:
        """
        Check if `claim` is novel relative to internal and external knowledge.
        """
        internal_matches = self._check_internal(claim, domain)
        external_matches: list = []
        contradictions:   list = []

        if search and self.search_fn and len(internal_matches) < 2:
            external_matches, contradictions = self._check_external(
                claim, domain, context
            )

        # Compute status
        n_internal = len(internal_matches)
        n_external = len(external_matches)
        n_contra   = len(contradictions)

        if n_internal >= 3 or n_external >= 2:
            status     = "known"
            confidence = min(0.95, 0.5 + 0.1 * (n_internal + n_external))
        elif n_internal >= 1 or n_external >= 1:
            status     = "similar"
            confidence = 0.6
        elif n_contra >= 1:
            status     = "similar"   # contradicting evidence = related work exists
            confidence = 0.65
        elif not search or self.search_fn is None:
            status     = "inconclusive"
            confidence = 0.4
        else:
            status     = "novel"
            confidence = 0.7

        recommendation = self._make_recommendation(
            status, n_internal, n_external, n_contra, domain
        )

        return NoveltyResult(
            status=status,
            confidence=confidence,
            internal_matches=internal_matches,
            external_matches=external_matches,
            contradictions=contradictions,
            recommendation=recommendation,
        )

    def _check_internal(self, claim: str, domain: str) -> list:
        matches = []

        # Episodic memory
        similar_runs = self.episodic.query_similar(claim, top_n=5)
        for run, score in similar_runs:
            if score > self.sim_thresh:
                matches.append({
                    "source":   "episodic",
                    "text":     run.question[:100],
                    "score":    round(score, 3),
                    "verified": run.verified,
                })

        # Technique library
        techs = self.library._data.get("techniques", {})
        claim_words = set(_tokenise(claim))
        for t in techs.values():
            if t.get("domain") != domain and domain:
                continue
            tech_words = set(_tokenise(
                t.get("name","") + " " + t.get("description","")
            ))
            if not claim_words or not tech_words:
                continue
            overlap = len(claim_words & tech_words) / max(1, len(claim_words))
            if overlap > self.sim_thresh:
                matches.append({
                    "source": "technique_library",
                    "text":   t.get("name", ""),
                    "score":  round(overlap, 3),
                })

        return matches[:5]

    def _check_external(
        self, claim: str, domain: str, context: str
    ) -> tuple[list, list]:
        """
        Search externally for related work. Returns (matches, contradictions).
        """
        query = _formulate_search_query(claim, domain, context)
        try:
            results = self.search_fn(query)
        except Exception:
            return [], []

        if not results:
            return [], []

        matches:       list = []
        contradictions: list = []

        contradiction_markers = [
            "disproves", "contradicts", "incorrect", "wrong",
            "no evidence", "fails to", "inconsistent with",
        ]
        support_markers = [
            "confirms", "consistent with", "agrees", "verified",
            "demonstrates", "shows", "finds", "establishes",
        ]

        claim_lower = claim.lower()
        for r in results[:6]:
            snippet = (r.get("snippet") or r.get("description") or "").lower()
            title   = (r.get("title") or "").lower()
            text    = snippet + " " + title

            # Similarity to the claim
            sim = _word_overlap(claim_lower, text)
            if sim < 0.1:
                continue

            entry = {
                "title":   r.get("title", "")[:100],
                "snippet": (r.get("snippet") or "")[:200],
                "url":     r.get("url", ""),
                "sim":     round(sim, 3),
            }

            is_contradiction = any(m in text for m in contradiction_markers)
            is_support       = any(m in text for m in support_markers)

            if is_contradiction:
                contradictions.append(entry)
            elif is_support or sim > 0.25:
                matches.append(entry)

        return matches, contradictions

    def _make_recommendation(
        self,
        status:      str,
        n_internal:  int,
        n_external:  int,
        n_contra:    int,
        domain:      str,
    ) -> str:
        if status == "known":
            return (
                "This result is consistent with existing work. "
                "Record it as a verified replication and extract "
                "the technique for the library."
            )
        elif status == "similar":
            if n_contra > 0:
                return (
                    "Related work exists but contradicts this result. "
                    "Investigate the contradiction before accepting. "
                    "Flag as a high-priority obligation."
                )
            return (
                "Similar work exists. Identify the precise difference "
                "from prior results and strengthen verification. "
                "Consider whether this extends or refines existing knowledge."
            )
        elif status == "novel":
            return (
                "No matching prior work found. Increase verification rigour "
                "before treating as established. Consider bootstrapping "
                "additional verification methods for this domain. "
                "This is a candidate for external publication."
            )
        else:
            return (
                "Could not determine novelty without search. "
                "Enable web search or expand the internal knowledge base."
            )


# ============================================================
# Knowledge extraction from search results
# ============================================================

@dataclass
class DomainKnowledge:
    """
    Knowledge extracted from external sources about a domain.
    Used to bootstrap verification and technique generation.
    """
    domain:               str
    verification_methods: list[str]   # how to verify claims in this domain
    key_techniques:       list[str]   # core reasoning approaches
    key_vocabulary:       list[str]   # important terms
    sources:              list[str]   # where this came from
    raw_snippets:         list[str]   # raw text for the code generator to use

    def to_prompt_fragment(self) -> str:
        """Format as context for technique/verifier generation."""
        lines = [
            f"Domain knowledge for '{self.domain}':",
            "",
            "Verification methods used in this domain:",
        ]
        for v in self.verification_methods[:5]:
            lines.append(f"  - {v}")
        lines.append("")
        lines.append("Core reasoning techniques:")
        for t in self.key_techniques[:5]:
            lines.append(f"  - {t}")
        if self.key_vocabulary:
            lines.append("")
            lines.append(f"Key vocabulary: {', '.join(self.key_vocabulary[:10])}")
        return "\n".join(lines)


def extract_domain_knowledge(
    domain:     str,
    search_fn:  Callable,
    n_queries:  int = 3,
) -> DomainKnowledge:
    """
    Search for methodology and verification practices in a domain.
    Returns a DomainKnowledge object populated from search results.
    """
    queries = [
        f"verification methods {domain} research",
        f"reasoning techniques {domain} methodology",
        f"how to validate results in {domain}",
    ][:n_queries]

    all_snippets:     list[str] = []
    all_sources:      list[str] = []
    verification_raw: list[str] = []
    technique_raw:    list[str] = []
    vocab_raw:        list[str] = []

    for query in queries:
        try:
            results = search_fn(query)
        except Exception:
            continue
        for r in (results or [])[:4]:
            snippet = r.get("snippet") or r.get("description") or ""
            url     = r.get("url", "")
            if snippet:
                all_snippets.append(snippet)
            if url:
                all_sources.append(url)

    # Extract verification methods (sentences containing verification markers)
    for snippet in all_snippets:
        for sentence in re.split(r"[.!?]", snippet):
            sentence = sentence.strip()
            if not sentence:
                continue
            lower = sentence.lower()
            if any(m in lower for m in [
                "verif", "valid", "test", "check", "assess", "measur",
                "evaluat", "statistic", "significance", "p-value", "control",
            ]):
                verification_raw.append(sentence[:150])
            elif any(m in lower for m in [
                "method", "approach", "technique", "algorithm", "model",
                "framework", "analysis", "procedure",
            ]):
                technique_raw.append(sentence[:150])

    # Extract key vocabulary (domain-specific capitalised terms or frequent terms)
    all_text = " ".join(all_snippets)
    words    = re.findall(r"\b[A-Z][a-z]{3,}\b|\b[a-z]{5,}\b", all_text)
    word_freq: dict[str, int] = {}
    for w in words:
        w = w.lower()
        if w not in _COMMON_WORDS:
            word_freq[w] = word_freq.get(w, 0) + 1
    top_vocab = sorted(word_freq, key=lambda w: word_freq[w], reverse=True)[:15]

    return DomainKnowledge(
        domain=domain,
        verification_methods=list(dict.fromkeys(verification_raw[:8])),
        key_techniques=list(dict.fromkeys(technique_raw[:8])),
        key_vocabulary=top_vocab,
        sources=all_sources[:5],
        raw_snippets=all_snippets[:8],
    )


# ============================================================
# Verification bootstrapper
# ============================================================

class VerificationBootstrapper:
    """
    Generates domain-specific verification tools from external knowledge.

    When the system encounters a new domain with no verification tools,
    this bootstrapper:
      1. Searches for verification methodology in that domain
      2. Extracts verification approaches from the results
      3. Generates Python verification functions using the CodeGenerator
      4. Tests them in the sandbox
      5. Adds them to the technique library as verification techniques

    The system builds its own verification infrastructure from scratch
    for any new domain.
    """

    def __init__(
        self,
        search_fn:         Callable,
        code_generator,
        technique_library,
        llm_chat_fn:       Optional[Callable] = None,
        verbose:           bool = True,
    ):
        self.search_fn  = search_fn
        self.generator  = code_generator
        self.library    = technique_library
        self.llm        = llm_chat_fn
        self.verbose    = verbose

    def bootstrap_domain(
        self,
        domain:         str,
        n_techniques:   int = 3,
        n_verifiers:    int = 2,
    ) -> dict:
        """
        Full bootstrapping cycle for a new domain.
        Returns a summary of what was generated and added.
        """
        if self.verbose:
            print(f"\n[bootstrapper] Bootstrapping verification for '{domain}'...")

        # 1. Extract domain knowledge from search
        knowledge = extract_domain_knowledge(domain, self.search_fn)
        if self.verbose:
            print(f"  Found {len(knowledge.verification_methods)} verification methods")
            print(f"  Found {len(knowledge.key_techniques)} technique descriptions")

        added_techniques = 0
        added_verifiers  = 0

        # 2. Generate techniques from domain knowledge
        if self.llm:
            for i in range(min(n_techniques, len(knowledge.key_techniques))):
                tech = self._generate_technique_from_knowledge(
                    domain, knowledge, i
                )
                if tech:
                    added_techniques += 1
                    if self.verbose:
                        print(f"  Added technique: '{tech.get('name', '?')}'")

        # 3. Generate verification functions from domain knowledge
        if self.llm:
            for i in range(min(n_verifiers, len(knowledge.verification_methods))):
                verifier = self._generate_verifier_from_knowledge(
                    domain, knowledge, i
                )
                if verifier:
                    added_verifiers += 1
                    if self.verbose:
                        print(f"  Added verifier for: "
                              f"'{knowledge.verification_methods[i][:50]}'")

        result = {
            "domain":             domain,
            "n_search_snippets":  len(knowledge.raw_snippets),
            "n_verif_methods":    len(knowledge.verification_methods),
            "added_techniques":   added_techniques,
            "added_verifiers":    added_verifiers,
            "key_vocab":          knowledge.key_vocabulary[:8],
            "sources":            knowledge.sources[:3],
        }

        if self.verbose:
            print(f"  Bootstrap complete: {added_techniques} techniques, "
                  f"{added_verifiers} verifiers added.")

        return result

    def _generate_technique_from_knowledge(
        self, domain: str, knowledge: DomainKnowledge, idx: int,
    ) -> Optional[dict]:
        """Generate one technique dict from extracted domain knowledge."""
        if not knowledge.key_techniques:
            return None

        technique_hint = knowledge.key_techniques[idx % len(knowledge.key_techniques)]

        system = (
            "You generate technique library entries for a reasoning scaffold. "
            "Return a Python dict with: name, description, when_to_use, example_text. "
            "The technique must be specific and actionable. "
            "Return ONLY the Python dict literal."
        )
        user = (
            f"Domain: {domain}\n\n"
            f"{knowledge.to_prompt_fragment()}\n\n"
            f"Generate a technique based on this description:\n"
            f"  '{technique_hint}'\n\n"
            f"The technique should include concrete steps in example_text "
            f"that would actually help solve problems in this domain."
        )

        try:
            code = self.llm(system, user).strip()
            code = _clean_code_block(code)
        except Exception:
            return None

        # Validate in sandbox
        from code_generator import CodeSandbox
        sandbox = CodeSandbox()
        check   = (
            f"_d = {code}\n"
            f"assert isinstance(_d, dict)\n"
            f"assert all(k in _d for k in "
            f"['name','description','when_to_use','example_text'])\n"
        )
        result = sandbox.execute(check)
        if not result.success:
            return None

        # Extract and add to library
        try:
            namespace: dict = {}
            exec(f"_d = {code}", {"__builtins__": {
                "True": True, "False": False, "None": None,
            }}, namespace)
            t = namespace.get("_d")
            if not isinstance(t, dict):
                return None
            self.library.record_success(
                name=t["name"], description=t["description"],
                when_to_use=t["when_to_use"], example_text=t["example_text"],
                domain=domain, run_id=f"bootstrap-{domain}-{idx}",
            )
            return t
        except Exception:
            return None

    def _generate_verifier_from_knowledge(
        self, domain: str, knowledge: DomainKnowledge, idx: int,
    ) -> Optional[dict]:
        """Generate a domain-specific verification function."""
        if not knowledge.verification_methods:
            return None

        verif_hint = knowledge.verification_methods[idx % len(knowledge.verification_methods)]

        system = (
            "You generate Python verification functions for a reasoning scaffold. "
            "The function must:\n"
            "  - Return {verified: bool, note: str}\n"
            "  - Use only: numpy (as np), sympy, math, basic Python\n"
            "  - Be under 40 lines\n"
            "  - Handle exceptions gracefully\n"
            "Return ONLY the Python function."
        )
        user = (
            f"Domain: {domain}\n\n"
            f"Verification method to implement:\n  '{verif_hint}'\n\n"
            f"Domain context:\n{knowledge.to_prompt_fragment()}\n\n"
            f"Generate a Python verification function appropriate for "
            f"this domain and method. The function should check whether "
            f"a claim or result is valid by this method."
        )

        try:
            code = self.llm(system, user).strip()
            code = _clean_code_block(code)
        except Exception:
            return None

        # Test in sandbox
        from code_generator import CodeSandbox
        import ast as _ast
        sandbox = CodeSandbox()
        try:
            tree = _ast.parse(code)
            func_names = [n.name for n in _ast.walk(tree)
                          if isinstance(n, _ast.FunctionDef)]
            if not func_names:
                return None
            test_call = f"{func_names[0]}()"
        except Exception:
            return None

        result = sandbox.execute(code, test_call)
        if not result.success:
            return None

        return {"domain": domain, "method": verif_hint, "function_name": func_names[0], "code": code}


# ============================================================
# Domain Bootstrap Manager — top-level coordinator
# ============================================================

class DomainBootstrapManager:
    """
    Detects when a new domain is encountered and triggers bootstrapping.

    Integrates with the introspection module to identify domains with
    low maturity, then uses VerificationBootstrapper to build tools.
    """

    def __init__(
        self,
        bootstrapper:        VerificationBootstrapper,
        novelty_detector:    NoveltyDetector,
        maturity_threshold:  str = "early",   # bootstrap if maturity <= this
        log_path:            str = "./bootstrap_log.json",
        verbose:             bool = True,
    ):
        self.bootstrapper  = bootstrapper
        self.detector      = novelty_detector
        self.threshold     = maturity_threshold
        self.log_path      = log_path
        self.verbose       = verbose
        self._log:         list[dict] = self._load()
        self._bootstrapped: set[str]  = {e["domain"] for e in self._log}

    def check_and_bootstrap(
        self,
        domain:     str,
        self_model = None,
        force:      bool = False,
    ) -> Optional[dict]:
        """
        Check if `domain` needs bootstrapping and do it if so.
        Skips domains already bootstrapped unless force=True.
        """
        if domain in self._bootstrapped and not force:
            return None

        # Check maturity via self_model if available
        if self_model is not None and not force:
            self_model.compute_domain_profiles()
            profile = self_model._profiles.get(domain)
            if profile and profile.overall_maturity not in ("early",):
                return None   # already mature enough

        if self.verbose:
            print(f"[bootstrap_manager] New domain detected: '{domain}'")

        result = self.bootstrapper.bootstrap_domain(domain)
        result["bootstrapped_at"] = time.time()
        self._log.append(result)
        self._bootstrapped.add(domain)
        self._save()
        return result

    def check_finding_novelty(
        self,
        claim:   str,
        domain:  str,
        context: str = "",
        search:  bool = True,
    ) -> NoveltyResult:
        """
        Check whether a finding is genuinely new.
        Convenience wrapper around NoveltyDetector.
        """
        return self.detector.check(claim, domain, context, search)

    # ---- Persistence ----

    def _load(self) -> list[dict]:
        if not _os_path_exists(self.log_path):
            return []
        try:
            with open(self.log_path) as f:
                return json.load(f).get("bootstrapped", [])
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self) -> None:
        import os
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump({"bootstrapped": self._log}, f, indent=2)


# ============================================================
# Helpers
# ============================================================

_COMMON_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has",
    "him", "his", "how", "man", "new", "now", "old", "see", "two",
    "way", "who", "its", "did", "may", "said", "each", "she", "use",
    "this", "that", "with", "they", "been", "from", "more", "have",
    "their", "would", "about", "which", "there", "when", "make",
    "like", "time", "just", "know", "take", "into", "year", "your",
    "good", "some", "could", "them", "than", "then", "look", "only",
    "come", "over", "also", "back", "after", "most", "very", "what",
    "even", "here", "through", "used", "such", "well", "because",
    "does", "being", "these", "those", "will", "other", "data",
    "research", "study", "paper", "journal", "results", "show",
    "using", "based", "found", "analysis", "between",
}


def _tokenise(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]{4,}", text.lower())
            if t not in _COMMON_WORDS]


def _word_overlap(a: str, b: str) -> float:
    wa = set(_tokenise(a))
    wb = set(_tokenise(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _formulate_search_query(claim: str, domain: str, context: str) -> str:
    """
    Formulate a concise, effective search query from a claim.
    """
    # Extract the most distinctive words from the claim
    tokens = _tokenise(claim)
    # Prefer longer, more distinctive words
    tokens = sorted(set(tokens), key=len, reverse=True)[:5]
    query  = " ".join(tokens)
    if domain:
        query = f"{domain} {query}"
    return query[:150]


def _clean_code_block(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        inner = [l for l in lines[1:] if not l.startswith("```")]
        code  = "\n".join(inner).strip()
    return code


def _os_path_exists(path: str) -> bool:
    import os
    return os.path.exists(path)
