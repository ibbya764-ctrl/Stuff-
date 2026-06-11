# Mac Setup Guide — 8GB MacBook

Quick setup for testing the scaffold on a Mac with 8GB RAM.
This works on both Apple Silicon (M1/M2/M3) and Intel Macs.
Apple Silicon is noticeably faster — check Activity Monitor → CPU
to see if you have an Apple M-chip.

---

## STEP 1 — Install Python

Go to https://www.python.org/downloads/
Download the macOS installer (3.11 or 3.12).
Run it. No special options needed.

Verify in Terminal (search "Terminal" in Spotlight):
```
python3 --version
```
Should show 3.11.x or 3.12.x

---

## STEP 2 — Install Ollama (runs the AI locally)

Go to https://ollama.com
Click Download → Download for Mac
Open the .dmg and drag Ollama to Applications. Run it.
You'll see a small llama icon in your menu bar.

Open Terminal and download the model:
```
ollama pull qwen2.5:7b
```

This downloads ~4.5GB. Only needs to happen once.

To start the local server (run this before using the scaffold):
```
ollama serve
```

Leave this Terminal window open while working.
Open a new Terminal tab/window for everything else.

---

## STEP 3 — Get your scaffold files onto the Mac

You need two sets of files in the same folder:

**Your existing files from Colab:**
Open your Colab notebook. In the Files panel (left sidebar),
download each of these:
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

**New files from this conversation** (download all from Claude):
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
- requirements.txt

Put ALL of them in one folder, e.g. ~/Desktop/scaffold/

---

## STEP 4 — Install Python packages

In Terminal, navigate to your folder:
```
cd ~/Desktop/scaffold
```

Install packages:
```
pip3 install numpy scipy sympy requests z3-solver
```

If pip3 isn't found, try:
```
python3 -m pip install numpy scipy sympy requests z3-solver
```

---

## STEP 5 — Set your API key (optional but useful)

Even with local model, keep a Claude API key as fallback.
Get one at https://console.anthropic.com (free tier available).

In Terminal:
```
export ANTHROPIC_API_KEY="your_key_here"
```

To make it permanent (so you don't have to set it every time),
add that line to ~/.zshrc:
```
echo 'export ANTHROPIC_API_KEY="your_key_here"' >> ~/.zshrc
source ~/.zshrc
```

---

## STEP 6 — Verify everything works

```
cd ~/Desktop/scaffold
python3 setup_check.py
```

Should show 11 green checkmarks and "Ready."

---

## STEP 7 — Update llm_chat in pipeline.py

Open pipeline.py in any text editor (TextEdit works, or download
VSCode from https://code.visualstudio.com for something nicer).

Find wherever `llm_chat` is defined and replace with:

```python
import requests
import os

def llm_chat(system: str, user: str) -> str:
    """
    Local Ollama on Mac 8GB — uses qwen2.5:7b.
    Falls back to Claude API if Ollama isn't running.
    """
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5:7b",
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
        # Fallback to Claude API
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("Ollama not running and no ANTHROPIC_API_KEY set.")
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

## STEP 8 — Run your pipeline

```
cd ~/Desktop/scaffold
python3 pipeline.py
```

Or however you normally run it from Colab — it's the same code,
just running locally now.

---

## Day-to-day on the Mac

1. Open Terminal
2. Run `ollama serve` (keep this window open)
3. Open a new Terminal tab: `cd ~/Desktop/scaffold`
4. Run your pipeline

---

## What works vs. what's slower on 8GB Mac

| Thing | Works? | Notes |
|---|---|---|
| All 11 scaffold modules | ✓ Full | Python only, no GPU needed |
| Qwen2.5-7B local inference | ✓ Good | ~20-30 tok/s on M-chip |
| Curiosity engine overnight | ✓ Yes | Leave it running |
| Structure mapper / GUE | ✓ Full | CPU only, fast enough |
| Qwen2.5-14B | ✗ Too big | Needs 9GB+ |
| Running without Ollama | ✓ Via API | Uses Claude API key |

The 7B model is about 80% as capable as 14B for structured output.
Good enough for testing and development.
When you switch to the RTX 4060 Ti 16GB, just change
`qwen2.5:7b` → `qwen2.5:14b` in the llm_chat function.
Everything else stays exactly the same.

---

## Common issues on Mac

**"command not found: python3"**
→ Reinstall Python from python.org, make sure to check "Add to PATH"

**"ollama: command not found"**
→ Make sure Ollama app is running (llama in menu bar)
→ Try restarting Terminal after launching Ollama

**"Connection refused" when calling llm_chat**
→ `ollama serve` isn't running
→ Open a new Terminal and run it

**"No module named X"**
→ All scaffold files need to be in the same folder
→ Run `pip3 install numpy scipy sympy requests` again

**Slow inference (Intel Mac)**
→ Normal — Intel uses CPU only, expect 5-10 tok/s
→ Still works, just be patient
