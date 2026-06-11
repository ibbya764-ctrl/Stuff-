"""
deep_research.py
================

Multi-step research pipeline for questions that need more than a
single LLM pass to answer properly.

Triggered when:
  - ComputeRouter classifies the question as DEEP
  - Question contains research signals ("latest", "current state of",
    "what do we know about", "recent studies", "review")
  - Explicitly requested ("research this", "look into this deeply")

Process:
  1. Decompose the question into 3-5 focused sub-questions
  2. Search academic databases for each in parallel
  3. Fetch and read the most relevant results
  4. Synthesise findings into a coherent understanding
  5. Verify the synthesis against what was found
  6. Return synthesis + all sources found

The synthesis goes through the normal Language module for
natural expression, but is grounded in actual retrieved content.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable


RESEARCH_SIGNALS = [
    "latest research", "current state", "recent studies",
    "what do we know", "scientific consensus", "evidence for",
    "review of", "look into", "research this", "find out",
    "search for", "investigate", "explore the literature",
]


@dataclass
class ResearchResult:
    question:       str
    sub_questions:  list
    sources:        list   # [{title, summary, source, url}]
    synthesis:      str    # synthesised understanding
    confidence:     str
    verified:       bool
    elapsed:        float
    n_sources:      int


class DeepResearch:
    """
    Multi-step research pipeline.

    When a question warrants deep research, this runs a structured
    process: decompose → search → read → synthesise → verify.
    Each step uses the LLM plus the specialist search databases.
    """

    def __init__(
        self,
        llm_chat_fn:         Callable,
        knowledge_orchestrator = None,
        verbose:             bool = True,
    ):
        self.llm       = llm_chat_fn
        self.knowledge = knowledge_orchestrator
        self.verbose   = verbose

    def needs_research(self, question: str) -> bool:
        lower = question.lower()
        return any(signal in lower for signal in RESEARCH_SIGNALS)

    def research(
        self,
        question: str,
        domain:   str = "general",
        max_sources: int = 8,
        time_budget_sec: int = 60,
    ) -> ResearchResult:
        """
        Run the full research pipeline on a question.
        Returns a ResearchResult with synthesis and sources.
        """
        t0 = time.time()
        if self.verbose:
            print(f"[research] Starting deep research: {question[:60]}")

        # Step 1: Decompose into sub-questions
        sub_questions = self._decompose(question, domain)
        if self.verbose:
            print(f"[research] {len(sub_questions)} sub-questions")

        # Step 2: Search all databases in parallel
        all_sources = self._parallel_search(sub_questions, domain, max_sources)
        if self.verbose:
            print(f"[research] {len(all_sources)} sources found")

        # Check time budget
        if time.time() - t0 > time_budget_sec * 0.6:
            # Synthesise with what we have
            pass

        # Step 3: Synthesise
        synthesis = self._synthesise(question, all_sources, domain)

        # Step 4: Verify synthesis plausibility
        verified = self._verify_synthesis(synthesis, all_sources)

        elapsed = round(time.time() - t0, 1)
        if self.verbose:
            print(f"[research] Complete in {elapsed}s, verified={verified}")

        # Store sources in knowledge base
        if self.knowledge and all_sources:
            for src in all_sources[:5]:
                fact = f"{src.get('title','')}: {src.get('summary','')[:150]}"
                self.knowledge.store_learned_fact(
                    fact, domain, src.get("source","research"), confidence=0.7
                )

        return ResearchResult(
            question=question,
            sub_questions=sub_questions,
            sources=all_sources,
            synthesis=synthesis,
            confidence="MODERATE" if verified else "LOW",
            verified=verified,
            elapsed=elapsed,
            n_sources=len(all_sources),
        )

    def _decompose(self, question: str, domain: str) -> list[str]:
        """Break the question into focused search queries."""
        system = (
            "You decompose a research question into 3-5 specific sub-questions "
            "suitable for academic database search. Output one per line, no numbering."
        )
        user = f"Question: {question}\nDomain: {domain}\nSub-questions:"
        try:
            response = self.llm(system, user).strip()
            lines = [l.strip() for l in response.split("\n") if l.strip() and len(l.strip()) > 10]
            return lines[:5] if lines else [question]
        except Exception:
            return [question]

    def _parallel_search(
        self, queries: list[str], domain: str, max_total: int
    ) -> list[dict]:
        """Search all databases for all queries in parallel threads."""
        if not self.knowledge:
            return []

        results = []
        lock = threading.Lock()
        per_query = max(2, max_total // max(1, len(queries)))

        def search_one(query):
            try:
                found = self.knowledge.search_specialist_literature(
                    query, domain, n=per_query
                )
                with lock:
                    results.extend(found)
            except Exception:
                pass

        threads = [threading.Thread(target=search_one, args=(q,), daemon=True)
                   for q in queries[:5]]
        for t in threads: t.start()
        for t in threads: t.join(timeout=20)

        # Deduplicate by title
        seen, unique = set(), []
        for r in results:
            title = r.get("title","")
            if title and title not in seen:
                seen.add(title)
                unique.append(r)
        return unique[:max_total]

    def _synthesise(self, question: str, sources: list[dict], domain: str) -> str:
        """Synthesise sources into a coherent understanding."""
        if not sources:
            return "No relevant sources found for this question."

        sources_text = "\n\n".join(
            f"[{i+1}] {s.get('title','')}\n{s.get('summary','')[:200]}"
            for i, s in enumerate(sources[:6])
        )
        system = (
            "You synthesise research findings into a coherent, precise understanding. "
            "Ground every claim in the sources provided. "
            "Note where sources agree or disagree. "
            "Be honest about uncertainty and gaps. "
            "Prose only — no bullet points or headers."
        )
        user = (
            f"Question: {question}\n\n"
            f"Sources:\n{sources_text}\n\n"
            f"Synthesise what these sources tell us about this question."
        )
        try:
            return self.llm(system, user).strip()
        except Exception:
            summaries = [s.get("summary","")[:100] for s in sources[:3]]
            return " ".join(summaries)

    def _verify_synthesis(self, synthesis: str, sources: list[dict]) -> bool:
        """Check that synthesis is grounded in what was actually found."""
        if not sources or not synthesis:
            return False
        # Simple check: do key terms from sources appear in synthesis?
        all_source_words = set()
        for s in sources[:5]:
            text = (s.get("title","") + " " + s.get("summary","")).lower()
            all_source_words.update(w for w in text.split() if len(w) > 5)
        synthesis_words = set(synthesis.lower().split())
        overlap = len(all_source_words & synthesis_words)
        return overlap > 10   # meaningful overlap with source material


def should_research(question: str, level: str = "light") -> bool:
    """Quick check if a question warrants deep research."""
    if level == "deep":
        return True
    lower = question.lower()
    return any(signal in lower for signal in RESEARCH_SIGNALS)
