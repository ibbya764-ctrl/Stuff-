# Complete Setup Guide — RTX 4060 Ti 16GB

## What you need to do (in order)

---

### STEP 1 — Install Python

Go to https://www.python.org/downloads/
Download Python 3.11 or 3.12. During install, tick "Add Python to PATH".

Verify in terminal/command prompt:
```
python --version
```
Should show 3.11.x or 3.12.x

---

### STEP 2 — Create your project folder

Create a folder anywhere, e.g. `C:\ai_scaffold` or `~/ai_scaffold`

Put ALL the .py files from this project into that folder:

**Original files (from your existing Colab setup):**
- pipeline.py
- obligation_store.py
- technique_library.py
- reasoning_tools.py
- reasoning_tools_extended.py
- domains.py
- cross_branch.py
- counterexample_branches.py
- conversation.py
- router.py
- web_search_tool.py

**New files (from this conversation):**
- metacognition.py
- curiosity_engine.py
- technique_composer.py
- cross_domain_composer.py
- relational_graph.py
- graph_extractor.py
- structure_mapper.py
- gue_engine.py
- gue_domain_tester.py
- directed_graph.py
- hermitian_gue.py
- setup_check.py

---

### STEP 3 — Install Python packages

Open terminal/command prompt in your project folder and run:

```
pip install numpy scipy sympy requests z3-solver
```

---

### STEP 4 — Install Ollama (runs the AI model locally)

Go to https://ollama.com and click Download.
Install it like a normal program.

After installing, open a terminal and run:
```
ollama pull qwen2.5:14b
```

This downloads the Qwen2.5-14B model (~9GB). 
It only needs to do this once.

To start the local AI server (run this before using the scaffold):
```
ollama serve
```

Leave this terminal open while you work.

---

### STEP 5 — Set up your API key (for backup/heavy calls)

Even with local model, keep an Anthropic API key for when you need
the full Claude for complex reasoning.

Get one at: https://console.anthropic.com

**Windows:**
```
setx ANTHROPIC_API_KEY "your_key_here"
```
(restart terminal after this)

**Mac/Linux:**
```
export ANTHROPIC_API_KEY="your_key_here"
```

---

### STEP 6 — Verify everything works

In your project folder run:
```
python setup_check.py
```

You should see 11 green checkmarks and "Ready."

---

### STEP 7 — Update your pipeline's llm_chat function

In your `pipeline.py`, replace wherever you define `llm_chat` with:

```python
import requests
import os

def llm_chat(system: str, user: str) -> str:
    """
    Local Ollama inference on RTX 4060 Ti 16GB.
    Fast, free, runs overnight.
    """
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5:14b",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 2000,
                },
            },
            timeout=120,
        )
        return resp.json()["message"]["content"]
    except Exception:
        # Fallback to Claude API if local model is unavailable
        return _claude_fallback(system, user)

def _claude_fallback(system: str, user: str) -> str:
    """Fallback to Claude API when local model is down."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("No local Ollama and no ANTHROPIC_API_KEY set.")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
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

---

### STEP 8 — Add the new modules to a run

After your normal `result = pipeline.run(question)`, add:

```python
from metacognition import metacognitive_audit
from hermitian_gue import measure_hermitian_gue
from directed_graph import DirectedCoOccurrenceGraph
from gue_engine import SystemGUEMonitor

# Audit the result
audit = metacognitive_audit(
    result,
    llm_chat_fn=llm_chat,
    obligation_store=store,  # your existing ObligationStore instance
)

# Track GUE convergence
monitor = SystemGUEMonitor(log_path="./gue_log.json")
report = monitor.report(
    run_id=result.get("run_id", "run-1"),
    branch_scores=[b.get("total_score", 0) for b in result.get("all_branches", [])],
    obligation_store=store,
)
```

---

### Day-to-day workflow

1. Open terminal, run `ollama serve`
2. Open another terminal in your project folder
3. Run your pipeline normally
4. Let the curiosity engine run overnight (no API costs)
5. Check gue_log.json to track convergence

---

### What the 16GB gets you vs 8GB

| Model | VRAM needed | Quality | Speed |
|-------|------------|---------|-------|
| Qwen2.5-7B Q4 | ~5GB | Good | Fast (~80 tok/s) |
| **Qwen2.5-14B Q4** | **~9GB** | **Best for this** | **~45 tok/s** |
| Qwen2.5-14B FP16 | ~29GB | - | Too big |
| DeepSeek-R1-14B Q4 | ~9GB | Good for reasoning | ~40 tok/s |

The 16GB version means you have headroom. Try Qwen2.5-14B first.
If you want stronger reasoning specifically, also try:
```
ollama pull deepseek-r1:14b
```

---

### If something doesn't work

Common issues:

**"Ollama not found"** → restart terminal after installing Ollama

**"Connection refused"** → make sure `ollama serve` is running in another terminal

**"CUDA out of memory"** → model too big, use 7b instead of 14b

**Module import error** → make sure all .py files are in the same folder

**JSON parse error from LLM** → the model returned malformed JSON.
Qwen2.5-14B handles this well. If it persists, try lowering temperature to 0.3.
