# QUICKSTART — Getting the scaffold running

## Step 1: Install Python dependencies

```bash
pip install numpy scipy sympy requests
```

Optional but useful:
```bash
pip install z3-solver          # SMT verifier (improves verification quality)
pip install hypothesis         # Better counterexample search
```

## Step 2: Get an API key (for LLM calls)

The scaffold uses an LLM for branch generation, technique extraction,
and structure mapping. While you don't have a local GPU yet, use the
Anthropic API (Claude) — it's the same model I am.

1. Go to https://console.anthropic.com
2. Create an account and get an API key
3. Set it as an environment variable:

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "your_key_here"
```

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=your_key_here
```

**Mac/Linux:**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

Or create a `.env` file in the scaffold folder:
```
ANTHROPIC_API_KEY=your_key_here
```

## Step 3: Verify everything works

```bash
python setup_check.py
```

Expected output:
```
✓ relational_graph
✓ graph_extractor
... (all 11 modules)
✓ Claude API working
✓ All pipeline components operational
```

## Step 4: Set up the llm_chat function

In your pipeline, the LLM is called through a `llm_chat(system, user)`
function. Here's the one to use with Claude API:

```python
import requests
import os

def llm_chat(system: str, user: str) -> str:
    """Call Claude API. Drop this into pipeline.py."""
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2000,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=60,
    )
    return resp.json()["content"][0]["text"]
```

## Step 5: When your GPU arrives tomorrow

When the RTX 4060 Ti arrives, switch from API to local inference:

**Install Ollama** (https://ollama.com — one installer):
```bash
# After installing Ollama:
ollama pull qwen2.5:7b       # 7B model, fits on 8GB
# OR
ollama pull qwen2.5:14b      # 14B model, needs 16GB
```

**Swap the llm_chat function:**
```python
import requests

def llm_chat(system: str, user: str) -> str:
    """Local Ollama inference — no API costs, runs overnight."""
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": "qwen2.5:7b",   # or qwen2.5:14b if you have 16GB
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
    })
    return resp.json()["message"]["content"]
```

That's the only change needed — everything else in the scaffold works
the same.

## File structure

All scaffold files should be in the same directory:

```
your_folder/
├── pipeline.py                  ← your existing pipeline (unchanged)
├── obligation_store.py          ← your existing obligation store
├── technique_library.py         ← your existing technique library
├── reasoning_tools.py           ← your existing tools
├── domains.py                   ← your existing domains
│
├── metacognition.py             ← NEW: challenger audit
├── curiosity_engine.py          ← NEW: self-directed exploration
├── technique_composer.py        ← NEW: within-domain composition
├── cross_domain_composer.py     ← NEW: cross-domain analogies
├── relational_graph.py          ← NEW: typed relational graphs
├── graph_extractor.py           ← NEW: LLM → graph extraction
├── structure_mapper.py          ← NEW: SME-style analogy
├── gue_engine.py                ← NEW: GUE distribution testing
├── gue_domain_tester.py         ← NEW: domain-aware GUE testing
├── directed_graph.py            ← NEW: Hermitian directed graph
├── hermitian_gue.py             ← NEW: GOE→GUE transition tracking
│
├── setup_check.py               ← run this first
└── requirements.txt
```

## Quick test of the new modules

```python
# Test the directed graph and Hermitian GUE
from directed_graph import DirectedCoOccurrenceGraph
from hermitian_gue import measure_hermitian_gue

g = DirectedCoOccurrenceGraph("./test_graph.json")

# Add some ordered runs (list = ordered, set = unordered)
g.update_from_run(["technique_A", "technique_B", "technique_C"])
g.update_from_run(["technique_A", "technique_B", "technique_D"])
g.update_from_run(["technique_B", "technique_C", "technique_D"])

print(g.summary())
# → shows imaginary_fraction > 0 (directional structure detected)

result = measure_hermitian_gue(g)
# → shows GUE/GOE score and whether directed structure is shifting toward GUE
```

## What to test first

Before adding new modules to your existing pipeline, run your
existing pipeline normally (ask it a physics question) and verify
it still works. Then add modules one at a time:

1. `metacognition` — add after `pipeline.run()` calls
2. `curiosity_engine` — run separately between sessions
3. `directed_graph` — replace your existing co-occurrence graph
4. `hermitian_gue` — run as a diagnostic after 20+ runs

Check the README files (METACOGNITION_AND_CURIOSITY_README.md,
TECHNIQUE_COMPOSER_README.md, etc.) for integration details.
