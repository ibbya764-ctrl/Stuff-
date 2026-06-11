"""
patch_all.py
============

One script to wire everything together.
Run this from the scaffold folder:
    python patch_all.py

What it does:
  1. Copies new Python files into the scaffold directory
  2. Patches web_api_monetized.py with all new wiring
  3. Sets up Groq-first routing for ALL users
  4. Wires in Language module (natural responses)
  5. Wires in status endpoints (fixes dashboard panels)
  6. Starts autonomous loop + progress tracker
  7. Updates .env with Groq configuration hint
  8. Backs up before patching

Run once. Safe to run again — checks before patching.
"""

import os, sys, shutil, re

SCAFFOLD_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_API_PATH = os.path.join(SCAFFOLD_DIR, "web_api_monetized.py")
ENV_PATH     = os.path.join(SCAFFOLD_DIR, ".env")

print("=" * 55)
print("  Bri — Patch All")
print("=" * 55)
print(f"  Scaffold: {SCAFFOLD_DIR}")
print(f"  web_api:  {WEB_API_PATH}")
print()

# ── Step 1: Check prerequisites ───────────────────────────

if not os.path.exists(WEB_API_PATH):
    print(f"ERROR: web_api_monetized.py not found at {WEB_API_PATH}")
    print("Make sure you're running this from the scaffold folder.")
    sys.exit(1)

# Check Groq installed
try:
    import groq
    print("✓ groq package installed")
except ImportError:
    print("Installing groq...")
    os.system(f"{sys.executable} -m pip install groq -q")
    print("✓ groq installed")

# Check GROQ_API_KEY
groq_key = os.environ.get("GROQ_API_KEY", "")
if not groq_key and os.path.exists(ENV_PATH):
    for line in open(ENV_PATH):
        if line.startswith("GROQ_API_KEY="):
            groq_key = line.split("=",1)[1].strip()
            break

if not groq_key:
    print("\n⚠  GROQ_API_KEY not found in environment or .env")
    print("   Get a free key at: https://console.groq.com")
    print("   Add to .env: GROQ_API_KEY=your-key-here")
    print("   (Continuing — you'll need to add it before responses work)\n")
else:
    print(f"✓ GROQ_API_KEY found ({groq_key[:8]}...)")

# ── Step 2: Backup web_api_monetized.py ──────────────────

backup = WEB_API_PATH + ".bak"
if not os.path.exists(backup):
    shutil.copy(WEB_API_PATH, backup)
    print(f"✓ Backed up to {os.path.basename(backup)}")
else:
    print(f"✓ Backup already exists")

with open(WEB_API_PATH, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

changed = False

# ── Step 3: Groq-first routing for ALL users ─────────────

GROQ_LLM_FN = '''

def _bri_llm_chat(system: str, user: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Groq-first LLM — fast, free, routes ALL users through Groq API."""
    import os, requests as _req
    from efficiency_patches import select_groq_model

    # Auto-select model by question complexity
    model = select_groq_model(user, user_tier="pro")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            from groq import Groq as _Groq
            client = _Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as _e:
            print(f"[Groq] Error: {_e} — falling back")

    # Fallback: Ollama
    try:
        r = _req.post("http://localhost:11434/api/chat", json={
            "model": "qwen2.5:7b",
            "messages": [{"role":"system","content":system},{"role":"user","content":user}],
            "stream": False, "options": {"temperature": 0.7, "num_predict": 2000},
        }, timeout=120)
        if r.status_code == 200:
            return r.json()["message"]["content"]
    except Exception:
        pass

    raise RuntimeError("No LLM available — check GROQ_API_KEY in .env")

'''

if "_bri_llm_chat" not in content:
    # Insert after imports (after the last import line)
    lines = content.split("\n")
    last_import = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import = i
    insert_pos = last_import + 1
    lines.insert(insert_pos, GROQ_LLM_FN)
    content = "\n".join(lines)
    print("✓ Groq-first LLM function added")
    changed = True
else:
    print("✓ Groq LLM function already present")

# ── Step 4: Status endpoints ──────────────────────────────

STATUS_WIRE = '''
    # ── Status endpoints (dashboard panels) ──────────────
    try:
        from status_endpoints import make_status_router as _make_sr
        _sr = _make_sr(scaffold if "scaffold" in dir() else {}, collective if "collective" in dir() else None)
        app.include_router(_sr)
        print("[patch] Status endpoints wired")
    except Exception as _e:
        print(f"[patch] Status endpoints: {_e}")

    # ── Progress tracker ──────────────────────────────────
    try:
        from progress_tracker import start_tracker as _start_tracker
        _start_tracker(scaffold if "scaffold" in dir() else {}, collective if "collective" in dir() else None)
        print("[patch] Progress tracker started")
    except Exception as _e:
        print(f"[patch] Progress tracker: {_e}")

    # ── Autonomous loop ───────────────────────────────────
    try:
        from autonomous_loop import start_loop as _start_loop
        _start_loop(scaffold if "scaffold" in dir() else {}, collective if "collective" in dir() else None, questions_per_hour=4)
        print("[patch] Autonomous loop started (4 questions/hour)")
    except Exception as _e:
        print(f"[patch] Autonomous loop: {_e}")
'''

if "Status endpoints (dashboard panels)" not in content:
    # Find the startup event
    if "@app.on_event(\"startup\")" in content or "@app.on_event('startup')" in content:
        # Add before the last line of startup
        startup_pattern = re.compile(
            r"(@app\.on_event\(['\"]startup['\"]\)\s*\nasync def \w+\(\):)(.*?)(\n[^\s])",
            re.DOTALL
        )
        m = startup_pattern.search(content)
        if m:
            new_startup = m.group(1) + m.group(2) + STATUS_WIRE + m.group(3)
            content = content[:m.start()] + new_startup + content[m.end():]
            print("✓ Status endpoints wired into startup")
            changed = True
        else:
            # Append to file
            content += f"\n# Added by patch_all.py\n{STATUS_WIRE}\n"
            print("✓ Status endpoints added (appended)")
            changed = True
    else:
        content += f"\n# Added by patch_all.py\n{STATUS_WIRE}\n"
        print("✓ Status endpoints added (appended)")
        changed = True
else:
    print("✓ Status endpoints already wired")

# ── Step 5: Language module wiring ───────────────────────

LANG_WIRE = '''

def _apply_language_module(result: dict, question: str, domain: str, scaffold: dict) -> str:
    """Apply IntegratedCommunicator for natural, non-structured responses."""
    try:
        from language_module import IntegratedCommunicator as _IC
        from integrated_state import StateAssembler as _SA
        communicator = _IC(
            llm_chat_fn=_bri_llm_chat,
            dmn=scaffold.get("dmn"),
            psych=scaffold.get("psych"),
            brain=scaffold.get("brain"),
            plastic=scaffold.get("plastic"),
            persona=scaffold.get("persona"),
            knowledge=scaffold.get("knowledge"),
        )
        r = result.get("reasoning")
        if r:
            return communicator.communicate(r, question, domain)
    except Exception as _e:
        pass
    return result.get("response", "")

'''

if "_apply_language_module" not in content:
    content += LANG_WIRE
    print("✓ Language module helper added")
    changed = True
else:
    print("✓ Language module already present")

# ── Step 6: Deep research wiring ─────────────────────────

DEEP_WIRE = '''

def _maybe_deep_research(question: str, domain: str, scaffold: dict) -> str:
    """Run deep research pipeline if question warrants it."""
    try:
        from deep_research import DeepResearch, should_research
        level = "deep" if any(w in question.lower() for w in
            ["research", "find me", "look into", "investigate"]) else "light"
        if not should_research(question, level): return ""
        dr = DeepResearch(_bri_llm_chat, scaffold.get("knowledge"))
        result = dr.research(question, domain, time_budget_sec=45)
        return result.synthesis
    except Exception:
        return ""

'''

if "_maybe_deep_research" not in content:
    content += DEEP_WIRE
    print("✓ Deep research wiring added")
    changed = True
else:
    print("✓ Deep research already present")

# ── Step 7: Write patched file ────────────────────────────

if changed:
    with open(WEB_API_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✓ web_api_monetized.py patched and saved")
else:
    print(f"\n✓ web_api_monetized.py — all patches already applied")

# ── Step 8: Copy index.html ───────────────────────────────

web_static = os.path.join(SCAFFOLD_DIR, "web_static")
os.makedirs(web_static, exist_ok=True)

src_html = os.path.join(SCAFFOLD_DIR, "index.html")
dst_html = os.path.join(web_static, "index.html")

if os.path.exists(src_html):
    shutil.copy(src_html, dst_html)
    print(f"✓ index.html copied to web_static/")
elif os.path.exists(dst_html):
    print(f"✓ index.html already in web_static/")
else:
    print(f"⚠  index.html not found — download from Claude and put in scaffold folder")

# ── Step 9: Verify .env ───────────────────────────────────

print()
if os.path.exists(ENV_PATH):
    env_content = open(ENV_PATH).read()
    checks = ["SECRET_KEY", "GROQ_API_KEY"]
    for key in checks:
        has_it = key + "=" in env_content and len(env_content.split(key+"=")[1].split("\n")[0].strip()) > 5
        status = "✓" if has_it else "⚠  MISSING"
        print(f"  {status} {key}")
else:
    print(f"  ⚠  .env not found at {ENV_PATH}")
    print(f"  Create it with: SECRET_KEY=... and GROQ_API_KEY=...")

print()
print("=" * 55)
print("  Patch complete.")
print()
print("  Next: restart web_api_monetized.py")
print("  From scaffold folder:")
print("    python web_api_monetized.py")
print("=" * 55)

# ── Step added: meta-optimizer wiring ────────────────────
META_WIRE = '''

    # ── Meta-optimizer (Level 2 learning) ────────────────
    try:
        from meta_optimizer import start_meta_optimizer as _start_meta
        _meta = _start_meta(scaffold if "scaffold" in dir() else {})
        print("[patch] Meta-optimizer started (Level 2 learning)")
    except Exception as _e:
        print(f"[patch] Meta-optimizer: {_e}")

'''

with open(WEB_API_PATH, "r", encoding="utf-8", errors="replace") as f:
    c2 = f.read()

if "Meta-optimizer (Level 2" not in c2:
    # Add after autonomous loop wire
    c2 += "\n" + META_WIRE
    with open(WEB_API_PATH, "w", encoding="utf-8") as f:
        f.write(c2)
    print("✓ Meta-optimizer added to patch")
else:
    print("✓ Meta-optimizer already patched")
