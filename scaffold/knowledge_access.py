"""
knowledge_access.py
===================

Gives the system access to knowledge beyond what the base model
contains — current events, specialist literature, and a growing
knowledge base that learns from every lookup.

Three components:

  RealtimeKnowledge      — accesses current information through web
                           search, news feeds, and Wikipedia. For
                           anything that changes or that the base model
                           might not have. Provides the answer to
                           "what is true right now?"

  SpecialistSearch       — searches academic and specialist databases:
                           ArXiv (physics, CS, math), PubMed (medicine,
                           biology), Semantic Scholar (general academic),
                           Wikipedia (structured knowledge). No API keys
                           needed for basic access.

  SpecialistKnowledgeBase — the accumulative layer. Every fact looked
                           up is stored. Every time it's looked up again
                           and confirmed, it strengthens. After enough
                           confirmed lookups from multiple sources, a
                           fact crystallises into stable background
                           knowledge — no longer needing to be searched.

                           This is how the system learns specialist
                           knowledge the way a specialist does: through
                           repeated verified exposure. Early on it looks
                           everything up. Over time, the most important
                           facts are background knowledge it just knows.

  KnowledgeOrchestrator  — coordinates all three. Before reasoning,
                           gathers relevant knowledge. During reasoning,
                           looks up specific claims. After reasoning,
                           stores what was learned.
"""

import os
import re
import json
import time
import math
import hashlib
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# KnowledgeNode — one fact in the specialist knowledge base
# ============================================================

@dataclass
class KnowledgeNode:
    """One fact the system has learned through repeated lookup."""

    node_id:      str
    fact:         str         # the factual claim
    domain:       str         # what domain this belongs to
    source:       str         # where it came from
    confidence:   float       # current confidence 0-1
    access_count: int         # how many times this has been accessed
    n_sources:    int         # how many distinct sources confirmed this
    first_seen:   float
    last_accessed: float
    crystallised: bool = False  # stable background knowledge

    # Crystallisation: accessed 5+ times from 2+ sources at 0.7+ confidence
    CRYSTALLISE_THRESHOLD_ACCESSES = 5
    CRYSTALLISE_THRESHOLD_SOURCES  = 2
    CRYSTALLISE_THRESHOLD_CONF     = 0.70

    def access(self, new_confidence: float, new_source: str = "") -> None:
        """Record another access. Strengthen if confirmed."""
        self.access_count  += 1
        self.last_accessed  = time.time()
        if new_confidence > 0:
            self.confidence = min(1.0, 0.7 * self.confidence + 0.3 * new_confidence)
        if new_source and new_source != self.source:
            self.n_sources = min(10, self.n_sources + 1)
        self._check_crystallisation()

    def _check_crystallisation(self) -> None:
        if (not self.crystallised
                and self.access_count  >= self.CRYSTALLISE_THRESHOLD_ACCESSES
                and self.n_sources     >= self.CRYSTALLISE_THRESHOLD_SOURCES
                and self.confidence    >= self.CRYSTALLISE_THRESHOLD_CONF):
            self.crystallised = True

    def relevance_score(self, query: str) -> float:
        """How relevant is this node to the query?"""
        q_words = set(re.findall(r"[a-z]{4,}", query.lower()))
        f_words = set(re.findall(r"[a-z]{4,}", self.fact.lower()))
        overlap  = len(q_words & f_words) / max(1, len(q_words | f_words))
        age_days = (time.time() - self.last_accessed) / 86400
        recency  = math.exp(-age_days / 30)   # half-life 30 days
        return overlap * 0.6 + recency * 0.2 + self.confidence * 0.2

    def to_dict(self) -> dict:
        return {
            "node_id":      self.node_id,
            "fact":         self.fact,
            "domain":       self.domain,
            "source":       self.source,
            "confidence":   round(self.confidence, 3),
            "access_count": self.access_count,
            "n_sources":    self.n_sources,
            "first_seen":   self.first_seen,
            "last_accessed": self.last_accessed,
            "crystallised": self.crystallised,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeNode":
        n = cls(
            node_id=d["node_id"], fact=d["fact"],
            domain=d.get("domain","general"), source=d.get("source",""),
            confidence=d.get("confidence", 0.5),
            access_count=d.get("access_count",1),
            n_sources=d.get("n_sources",1),
            first_seen=d.get("first_seen", time.time()),
            last_accessed=d.get("last_accessed", time.time()),
            crystallised=d.get("crystallised", False),
        )
        return n


# ============================================================
# SpecialistKnowledgeBase
# ============================================================

class SpecialistKnowledgeBase:
    """
    Accumulates specialist knowledge through repeated verified lookup.

    Early: looks everything up every time.
    Growing: frequently accessed facts strengthen.
    Mature: crystallised facts are background knowledge, injected
            automatically into reasoning context without searching.

    This mirrors how specialists learn: repeated exposure to the
    same important facts until they become internalized background.
    """

    def __init__(
        self,
        path:    str  = "./scaffold_data/specialist_knowledge.json",
        verbose: bool = True,
    ):
        self.path    = path
        self.verbose = verbose
        self._nodes: dict[str, KnowledgeNode] = {}
        self._load()

    def store(
        self,
        fact:       str,
        domain:     str,
        source:     str,
        confidence: float = 0.6,
    ) -> str:
        """Store a new fact or strengthen an existing one."""
        # Check if we already have this or something very similar
        existing = self._find_similar(fact, domain)
        if existing:
            existing.access(confidence, source)
            self._save()
            return existing.node_id

        node_id = hashlib.sha256(
            f"{fact[:80]}{domain}".encode()
        ).hexdigest()[:12]

        node = KnowledgeNode(
            node_id=node_id,
            fact=fact[:300],
            domain=domain,
            source=source[:100],
            confidence=confidence,
            access_count=1,
            n_sources=1,
            first_seen=time.time(),
            last_accessed=time.time(),
        )
        self._nodes[node_id] = node

        if self.verbose and node.crystallised:
            print(f"  [knowledge] Crystallised: {fact[:60]}")

        self._save()
        return node_id

    def lookup(
        self,
        query:    str,
        domain:   str = "",
        top_n:    int = 5,
    ) -> list[KnowledgeNode]:
        """Find the most relevant knowledge nodes for a query."""
        candidates = [
            n for n in self._nodes.values()
            if not domain or n.domain in (domain, "general")
        ]
        scored = [
            (n.relevance_score(query), n)
            for n in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [n for _, n in scored[:top_n] if scored[0][0] > 0.05]

        # Record access
        for n in results:
            n.last_accessed = time.time()
        if results:
            self._save()

        return results

    def crystallised_for_domain(self, domain: str) -> list[KnowledgeNode]:
        """Return all crystallised (stable background) knowledge for a domain."""
        return [
            n for n in self._nodes.values()
            if n.crystallised and n.domain in (domain, "general")
        ]

    def domain_profile(self, domain: str) -> dict:
        """How deep is our knowledge in a domain?"""
        nodes = [n for n in self._nodes.values()
                 if n.domain in (domain, "general")]
        crystallised = [n for n in nodes if n.crystallised]
        return {
            "total_facts":   len(nodes),
            "crystallised":  len(crystallised),
            "avg_confidence": (
                sum(n.confidence for n in nodes) / max(1, len(nodes))
            ),
            "depth": (
                "expert"     if len(crystallised) > 30 else
                "proficient" if len(crystallised) > 10 else
                "developing" if len(crystallised) > 3  else
                "beginning"  if len(nodes) > 0          else
                "none"
            ),
        }

    def format_background_knowledge(self, domain: str) -> str:
        """Format crystallised knowledge as pre-reasoning context."""
        stable = self.crystallised_for_domain(domain)
        if not stable:
            return ""
        lines = [f"[SPECIALIST KNOWLEDGE: {domain}]"]
        lines.append("  Established facts (verified through repeated lookup):")
        for n in stable[:8]:
            lines.append(f"    • {n.fact[:100]}")
        lines.append(
            f"  [{len(stable)} crystallised facts, "
            f"{self.domain_profile(domain)['total_facts']} total]"
        )
        lines.append("[/SPECIALIST KNOWLEDGE]")
        return "\n".join(lines)

    def _find_similar(self, fact: str, domain: str) -> Optional[KnowledgeNode]:
        """Find an existing node that's very similar to this fact."""
        q_words = set(re.findall(r"[a-z]{5,}", fact.lower()))
        for node in self._nodes.values():
            if node.domain != domain:
                continue
            n_words = set(re.findall(r"[a-z]{5,}", node.fact.lower()))
            overlap  = len(q_words & n_words) / max(1, len(q_words | n_words))
            if overlap > 0.7:
                return node
        return None

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for d in data.get("nodes", []):
                n = KnowledgeNode.from_dict(d)
                self._nodes[n.node_id] = n
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "nodes": [n.to_dict() for n in self._nodes.values()]
            }, f, indent=2)


# ============================================================
# SpecialistSearch — academic and structured databases
# ============================================================

class SpecialistSearch:
    """
    Searches specialist databases without API keys.

    ArXiv:           free, no key, covers physics/CS/math/bio
    PubMed:          free, covers medicine and life sciences
    Semantic Scholar: free, broad academic literature
    Wikipedia:       free, structured general knowledge
    """

    TIMEOUT = 8   # seconds per request

    def search_arxiv(
        self,
        query:       str,
        max_results: int = 5,
    ) -> list[dict]:
        """Search ArXiv for relevant academic papers."""
        q     = urllib.parse.quote(query)
        url   = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{q}&start=0&max_results={max_results}"
            f"&sortBy=relevance"
        )
        try:
            with urllib.request.urlopen(url, timeout=self.TIMEOUT) as r:
                content = r.read().decode("utf-8")

            results = []
            entries = re.findall(r"<entry>(.*?)</entry>", content, re.DOTALL)
            for entry in entries[:max_results]:
                title   = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                summary = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
                arxiv_id = re.search(r"<id>.*?abs/(.*?)</id>", entry)
                authors  = re.findall(r"<name>(.*?)</name>", entry)
                results.append({
                    "source":  "arxiv",
                    "id":      arxiv_id.group(1).strip() if arxiv_id else "",
                    "title":   title.group(1).strip() if title else "",
                    "summary": re.sub(r"\s+", " ",
                               summary.group(1).strip()) if summary else "",
                    "authors": authors[:3],
                    "url":     f"https://arxiv.org/abs/{arxiv_id.group(1).strip()}" if arxiv_id else "",
                })
            return results
        except Exception:
            return []

    def search_pubmed(
        self,
        query:       str,
        max_results: int = 5,
    ) -> list[dict]:
        """Search PubMed for medical/biological literature."""
        q    = urllib.parse.quote(query)
        url  = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pubmed&term={q}&retmax={max_results}&retmode=json"
        )
        try:
            with urllib.request.urlopen(url, timeout=self.TIMEOUT) as r:
                data = json.loads(r.read())
            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            # Fetch summaries
            id_str   = ",".join(ids[:max_results])
            sum_url  = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
                f"db=pubmed&id={id_str}&retmode=json"
            )
            with urllib.request.urlopen(sum_url, timeout=self.TIMEOUT) as r:
                sum_data = json.loads(r.read())

            results = []
            for pmid, doc in sum_data.get("result", {}).items():
                if pmid == "uids":
                    continue
                results.append({
                    "source":  "pubmed",
                    "id":      pmid,
                    "title":   doc.get("title", ""),
                    "summary": doc.get("sorttitle", ""),
                    "authors": [a.get("name","") for a in doc.get("authors", [])[:3]],
                    "year":    doc.get("pubdate","")[:4],
                    "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
            return results
        except Exception:
            return []

    def search_wikipedia(
        self,
        query:       str,
        max_results: int = 3,
    ) -> list[dict]:
        """Search Wikipedia for structured knowledge."""
        q   = urllib.parse.quote(query)
        url = (
            f"https://en.wikipedia.org/w/api.php?"
            f"action=query&list=search&srsearch={q}"
            f"&srlimit={max_results}&format=json"
        )
        try:
            with urllib.request.urlopen(url, timeout=self.TIMEOUT) as r:
                data = json.loads(r.read())

            results = []
            for item in data.get("query", {}).get("search", []):
                title   = item.get("title","")
                snippet = re.sub(r"<[^>]+>", "", item.get("snippet",""))
                results.append({
                    "source":  "wikipedia",
                    "title":   title,
                    "summary": snippet[:300],
                    "url":     f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}",
                })
            return results
        except Exception:
            return []

    def search_semantic_scholar(
        self,
        query:       str,
        max_results: int = 5,
    ) -> list[dict]:
        """Search Semantic Scholar for academic papers."""
        q   = urllib.parse.quote(query)
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search?"
            f"query={q}&limit={max_results}"
            f"&fields=title,abstract,year,authors,venue"
        )
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ScaffoldAI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as r:
                data = json.loads(r.read())
            results = []
            for paper in data.get("data", [])[:max_results]:
                results.append({
                    "source":  "semantic_scholar",
                    "id":      paper.get("paperId",""),
                    "title":   paper.get("title",""),
                    "summary": (paper.get("abstract","") or "")[:300],
                    "year":    paper.get("year",""),
                    "venue":   paper.get("venue",""),
                    "authors": [a.get("name","") for a in paper.get("authors",[])[:3]],
                })
            return results
        except Exception:
            return []

    def search_all(
        self,
        query:  str,
        domain: str = "general",
        n:      int = 3,
    ) -> list[dict]:
        """Search the most relevant databases for this domain."""
        domain_lower = domain.lower()
        results = []

        if any(d in domain_lower for d in ["physics","math","cs","computer"]):
            results += self.search_arxiv(query, n)

        if any(d in domain_lower for d in ["biology","medicine","medical","health"]):
            results += self.search_pubmed(query, n)

        if not results or any(d in domain_lower for d in ["general","history","philosophy","economics"]):
            results += self.search_wikipedia(query, n)

        if not results:
            results += self.search_semantic_scholar(query, n)

        return results[:n*2]


# ============================================================
# RealtimeKnowledge
# ============================================================

class RealtimeKnowledge:
    """
    Accesses current information through web search and Wikipedia.

    For anything the base model might not have:
    - Events after the training cutoff
    - Specific current facts (prices, statistics, who holds a position)
    - Verification of time-sensitive claims
    """

    def __init__(
        self,
        search_fn:   Optional[Callable] = None,
        specialist:  Optional[SpecialistSearch] = None,
    ):
        self.search_fn  = search_fn
        self.specialist = specialist or SpecialistSearch()

    def get(
        self,
        query:  str,
        domain: str = "general",
        n:      int = 3,
    ) -> dict:
        """
        Get current information for a query.
        Returns structured result with sources and confidence.
        """
        results = []

        # Try external search fn first
        if self.search_fn:
            try:
                raw = self.search_fn(query)
                for r in (raw or [])[:n]:
                    results.append({
                        "source":  "web",
                        "title":   r.get("title",""),
                        "content": r.get("snippet","")[:200],
                        "url":     r.get("url",""),
                    })
            except Exception:
                pass

        # Fall back to specialist search
        if not results:
            raw = self.specialist.search_all(query, domain, n)
            for r in raw[:n]:
                results.append({
                    "source":  r.get("source",""),
                    "title":   r.get("title",""),
                    "content": r.get("summary","")[:200],
                    "url":     r.get("url",""),
                })

        if not results:
            return {"query": query, "found": False, "results": []}

        return {
            "query":   query,
            "found":   True,
            "results": results,
            "summary": self._summarise(results, query),
        }

    def _summarise(self, results: list[dict], query: str) -> str:
        lines = []
        for r in results[:3]:
            if r.get("content"):
                lines.append(f"• [{r['source']}] {r['title']}: {r['content'][:120]}")
        return "\n".join(lines)


# ============================================================
# KnowledgeOrchestrator
# ============================================================

class KnowledgeOrchestrator:
    """
    Coordinates all knowledge sources.

    Before reasoning: gathers relevant background knowledge.
    During reasoning: looks up specific claims when needed.
    After reasoning: stores newly learned facts.

    The knowledge base grows richer with every session. Important
    facts crystallise into stable background knowledge over time.
    The system progressively becomes more knowledgeable in the
    domains it works in, without external curation.
    """

    def __init__(
        self,
        base_dir:    str  = "./scaffold_data",
        search_fn:   Optional[Callable] = None,
        verbose:     bool = True,
    ):
        self.verbose    = verbose
        self.realtime   = RealtimeKnowledge(search_fn=search_fn)
        self.specialist_search = SpecialistSearch()
        self.knowledge_base = SpecialistKnowledgeBase(
            path=os.path.join(base_dir, "specialist_knowledge.json"),
            verbose=verbose,
        )

    def pre_reasoning_context(
        self,
        question: str,
        domain:   str = "general",
    ) -> str:
        """
        Gather all relevant knowledge before reasoning begins.

        Returns formatted context string for injection into the Reasoner.
        Includes: crystallised background knowledge + recent search findings.
        """
        parts = []

        # 1. Crystallised background knowledge (no search needed)
        background = self.knowledge_base.format_background_knowledge(domain)
        if background:
            parts.append(background)

        # 2. Relevant stored facts
        stored = self.knowledge_base.lookup(question, domain, top_n=3)
        if stored:
            lines = ["[RELEVANT STORED KNOWLEDGE]"]
            for n in stored:
                status = "★ crystallised" if n.crystallised else f"confirmed {n.access_count}×"
                lines.append(f"  • {n.fact[:100]} ({status}, confidence {n.confidence:.2f})")
            lines.append("[/STORED]")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def lookup_during_reasoning(
        self,
        claim:  str,
        domain: str = "general",
    ) -> tuple[str, float]:
        """
        Look up a specific claim during reasoning.
        Returns (formatted result, confidence).
        Stores the result in the knowledge base.
        """
        # Check knowledge base first
        stored = self.knowledge_base.lookup(claim, domain, top_n=1)
        if stored and stored[0].confidence > 0.7:
            stored[0].access(stored[0].confidence)
            return (
                f"Known fact (confidence {stored[0].confidence:.2f}): "
                f"{stored[0].fact[:120]}",
                stored[0].confidence,
            )

        # Search for it
        result = self.realtime.get(claim, domain, n=3)
        if not result["found"]:
            return "No information found.", 0.0

        summary = result["summary"]
        confidence = 0.65   # web search is moderate confidence by default

        # Store the finding
        for r in result["results"][:2]:
            fact = f"{r['title']}: {r['content'][:150]}" if r.get("content") else r.get("title","")
            if fact:
                self.knowledge_base.store(fact, domain, r.get("source","web"), confidence)

        return summary, confidence

    def search_specialist_literature(
        self,
        query:  str,
        domain: str = "general",
        n:      int = 5,
    ) -> list[dict]:
        """
        Search specialist databases (ArXiv, PubMed, etc.) for a query.
        Stores relevant abstracts in the knowledge base.
        """
        results = self.specialist_search.search_all(query, domain, n)

        # Store abstracts in knowledge base
        for r in results:
            fact = f"{r['title']} ({r.get('year','?')}): {r.get('summary','')[:200]}"
            self.knowledge_base.store(
                fact, domain, r.get("source","academic"), confidence=0.75
            )

        if self.verbose and results:
            print(f"  [knowledge] Found {len(results)} specialist results for '{query[:40]}'")

        return results

    def store_learned_fact(
        self,
        fact:       str,
        domain:     str,
        source:     str  = "reasoning",
        confidence: float = 0.7,
    ) -> str:
        """Store a fact that was discovered during reasoning."""
        return self.knowledge_base.store(fact, domain, source, confidence)

    def domain_summary(self, domain: str) -> str:
        profile = self.knowledge_base.domain_profile(domain)
        return (
            f"Knowledge in {domain}: {profile['depth']} — "
            f"{profile['crystallised']} crystallised facts, "
            f"{profile['total_facts']} total "
            f"(avg confidence {profile['avg_confidence']:.2f})"
        )

    def status(self) -> str:
        all_nodes = list(self.knowledge_base._nodes.values())
        crystallised = sum(1 for n in all_nodes if n.crystallised)
        domains = {}
        for n in all_nodes:
            domains[n.domain] = domains.get(n.domain, 0) + 1
        lines = [
            "Knowledge Base:",
            f"  Total facts:   {len(all_nodes)}",
            f"  Crystallised:  {crystallised}",
            f"  Domains:       {', '.join(f'{d}({c})' for d,c in sorted(domains.items(), key=lambda x:-x[1])[:5])}",
        ]
        return "\n".join(lines)
