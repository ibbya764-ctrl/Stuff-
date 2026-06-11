# Getting it running on the MacBook today

## Step 1 — Install Python (if not already)

Go to https://www.python.org/downloads/
Download macOS 3.11 or 3.12. Run the installer.

Check it worked:
```
python3 --version
```

---

## Step 2 — Install Ollama and get the model

Go to https://ollama.com → Download for Mac → drag to Applications.
Launch Ollama (llama icon appears in menu bar).

Open Terminal and run:
```
ollama pull qwen2.5:7b
```
This downloads ~4.5GB. Takes 5-10 minutes on a decent connection.

While that downloads, do the next steps.

---

## Step 3 — Create your scaffold folder

```
mkdir ~/Desktop/scaffold
cd ~/Desktop/scaffold
```

---

## Step 4 — Get your existing pipeline files from Colab

Open your Colab notebook. In the Files panel (left sidebar, folder icon),
download each of these files:

  pipeline.py
  obligation_store.py
  technique_library.py
  reasoning_tools.py
  reasoning_tools_extended.py
  domains.py
  cross_branch.py
  counterexample_branches.py
  conversation.py
  router.py
  web_search_tool.py

Put all of them in ~/Desktop/scaffold/

---

## Step 5 — Download new modules from Claude

From this conversation, download all the files listed below.
Put them all in ~/Desktop/scaffold/ alongside the Colab files.

NEW MODULES:
  metacognition.py
  curiosity_engine.py
  technique_composer.py
  cross_domain_composer.py
  relational_graph.py
  graph_extractor.py
  structure_mapper.py
  gue_engine.py
  gue_domain_tester.py
  directed_graph.py
  hermitian_gue.py
  technique_embedder.py
  structural_branch_sampler.py
  memory_system.py
  introspection.py
  self_adjuster.py
  code_generator.py
  knowledge_seeker.py
  enhanced_pipeline.py
  scaffold_trainer.py

RUN FILES:
  mac_run.py          ← the main script to run
  setup_check.py      ← run this first to verify everything

---

## Step 6 — Install Python packages

```
cd ~/Desktop/scaffold
pip3 install numpy scipy sympy requests z3-solver
```

---

## Step 7 — Start Ollama server

In Terminal, run:
```
ollama serve
```
Leave this terminal window open.
Open a new terminal for everything else.

---

## Step 8 — Verify setup

```
cd ~/Desktop/scaffold
python3 setup_check.py
```

You should see green checkmarks for all modules.
If anything fails, check the error and re-download that file.

---

## Step 9 — Run the system

```
cd ~/Desktop/scaffold
python3 mac_run.py
```

This will:
- Initialise all 11 modules
- Run a physics question through the full pipeline
- Show every module firing (bootstrap, memory, introspection, novelty)
- Save everything to ./scaffold_data/
- Wait for you to enter more questions

---

## What works on the MacBook 8GB

| Thing | Works? |
|---|---|
| Full enhanced pipeline | ✓ Yes |
| All 20 new modules | ✓ Yes |
| Episodic memory / learning | ✓ Yes |
| Domain bootstrapping | ✓ Yes |
| Structure mapper / GUE | ✓ Yes |
| Training buffer accumulation | ✓ Yes |
| LoRA fine-tuning | ✗ No GPU — exports dataset instead |
| qwen2.5:7b on M-chip Mac | ✓ ~20-30 tokens/sec |
| qwen2.5:7b on Intel Mac | ✓ ~5-10 tokens/sec (slower but works) |

Fine-tuning falls back to export mode automatically.
The training dataset builds up in ./scaffold_data/training_dataset.jsonl
ready for the 4060 Ti tomorrow.

---

## What to check on the first run

After the first question, look for these in the output:

  [bootstrap_manager] New domain detected: 'physics_mond'
  → System detected a new domain and bootstrapped it

  [memory] ... similar past runs, ... relevant techniques
  → Memory is working (shows 0 on first run, grows after)

  [novelty] INCONCLUSIVE / NOVEL / SIMILAR
  → Novelty detection fired on the result

  Training buffer: N examples
  → Examples are being saved for fine-tuning

After 5+ runs on the same type of problem you'll see:
  [memory] 3 similar past runs, freshness=0.4
  [novelty] SIMILAR (confidence 60%)

That means the system is recognising its own prior work.

---

## When the 4060 Ti arrives

1. Install unsloth: pip install unsloth
2. The trainer will auto-detect it and switch from export_only to full LoRA
3. After 20 verified runs, fine-tuning fires automatically
4. The merged model can be loaded into Ollama as a custom model

The training dataset from the MacBook session will carry over —
all verified runs are saved to scaffold_data/ and will be used
in the first fine-tuning cycle.

---

## If Ollama is too slow on Intel Mac

Use the Claude API instead:
```
export ANTHROPIC_API_KEY="your_key_here"
python3 mac_run.py
```

mac_run.py automatically falls back to Claude API if Ollama isn't
running or doesn't respond. You'll see:
  [llm] Ollama not running — using Claude API

---

## Troubleshooting

"command not found: python3"
→ Reinstall Python from python.org

"No module named X"
→ All files need to be in the same folder
→ Run: pip3 install numpy scipy sympy requests

"Connection refused" when running
→ ollama serve needs to be running in a terminal

"Ollama running but slow"
→ Normal on Intel Mac. Each response takes 30-60 seconds.
→ Use Claude API for faster testing.

"ModuleNotFoundError: pipeline"
→ You need the original Colab files (pipeline.py etc.)
→ Download them from your Colab notebook Files panel
