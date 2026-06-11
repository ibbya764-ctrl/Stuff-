"""
web_search_tool.py
==================

Web search integration for the reasoning pipeline.

Adds a web_search tool to the tool registry so branches can
retrieve current information to ground factual claims.
Without this, the system can reason well but cannot look anything up —
making it unreliable for questions that require current knowledge.

Uses DuckDuckGo's free API (no key needed) as primary,
with a simple fallback.
"""

import re
import json
import urllib.request
import urllib.parse
from reasoning_tools import register_tool


@register_tool(
    name="web_search",
    description=(
        "Search the web for current information. Use this when a branch "
        "makes a factual claim that should be verifiable, when the question "
        "asks about recent work, or when you need to ground a claim in "
        "external sources rather than training data. Returns titles, "
        "snippets and URLs of top results."
    ),
    parameters={
        "query": {"type": "string",
                  "description": "Search query, 3-8 words works best"},
        "n_results": {"type": "integer", "default": 4},
    },
)
def web_search(query: str, n_results: int = 4) -> dict:
    """Search DuckDuckGo Lite and return structured results."""
    try:
        results = _ddg_search(query, n_results)
        if results:
            return {
                "ok": True,
                "query": query,
                "n_results": len(results),
                "results": results,
            }
    except Exception as e:
        return {"ok": False, "error": str(e), "query": query}

    return {"ok": True, "query": query, "n_results": 0, "results": []}


def _ddg_search(query: str, n: int) -> list:
    """DuckDuckGo Lite scrape — no API key required."""
    url = (
        "https://lite.duckduckgo.com/lite/?"
        + urllib.parse.urlencode({"q": query, "kl": "us-en"})
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ReasoningPipeline/1.0)"
        )
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=8) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    results = []
    # Extract result links and snippets from the lite HTML
    link_pat   = re.compile(r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>')
    snip_pat   = re.compile(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', re.DOTALL)

    links   = link_pat.findall(html)
    snippets = snip_pat.findall(html)

    for i in range(min(n, len(links))):
        href, title = links[i]
        snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
        results.append({
            "title":   title.strip(),
            "url":     href,
            "snippet": snippet[:300],
        })

    return results
