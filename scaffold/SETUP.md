# Setup Guide

## What you need

One folder with all the files listed below.
Then three commands. That's it.

---

## Step 1 — Install Python

https://www.python.org/downloads/
Download the macOS installer (3.11 or 3.12).

Check it worked — open Terminal, type:
    python3 --version

---

## Step 2 — Install Ollama

https://ollama.com → Download for Mac
Open the .dmg, drag to Applications, launch it.
(Small llama icon appears in the menu bar.)

Open Terminal and run:
    ollama pull qwen2.5:7b

This downloads ~4.5 GB. Takes a few minutes.

---

## Step 3 — Create your folder

    mkdir ~/Desktop/scaffold
    cd ~/Desktop/scaffold

---

## Step 4 — Files you need

Put ALL of these in ~/Desktop/scaffold/

---- FROM YOUR COLAB NOTEBOOK ----
Open Colab. Left sidebar → Files icon.
Download each of these:

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

---- FROM THIS CONVERSATION (download from Claude) ----

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
    continual_trainer.py
    mac_run.py
    setup_check.py
    requirements.txt

---

## Step 5 — Install Python packages

    cd ~/Desktop/scaffold
    pip3 install numpy scipy sympy requests z3-solver

---

## Step 6 — Start Ollama

In one terminal window, run:
    ollama serve

Leave that window open.
Open a NEW terminal window for the next steps.

---

## Step 7 — Check everything works

    cd ~/Desktop/scaffold
    python3 setup_check.py

You should see all green. If anything is missing, re-download that file.

---

## Step 8 — Run it

    python3 mac_run.py

It will run a first question automatically, then wait for yours.
Type any physics question. Or type 'explore' to let the system
choose its own question.

Type Ctrl+C to stop — it exports everything it learned to
./scaffold_data/training_data.jsonl automatically.

---

## What works on 8GB Mac

Everything except LoRA fine-tuning (needs GPU).

The system still:
- Runs full reasoning pipeline with all 11 modules
- Builds episodic memory across sessions
- Bootstraps new domains automatically
- Accumulates training data (saved for the 4060 Ti)
- Detects novelty in results
- Self-adjusts parameters
- Generates code to fix failure patterns

Fine-tuning runs in export-only mode — it saves
the training dataset and tells you how to load it
into the 4060 Ti when it arrives.

---

## When the 4060 Ti arrives

Install unsloth (makes training 2x faster, half the memory):
    pip install unsloth

Then everything is the same — just run mac_run.py as before.
The trainer auto-detects the GPU and switches from export mode
to live LoRA fine-tuning. Training triggers automatically after
every 5 verified runs.

---

## Common problems

"MISSING pipeline" or similar
→ Download those files from your Colab notebook

"Ollama not running"
→ Make sure you ran  ollama serve  in a separate terminal

"command not found: python3"
→ Re-install Python from python.org

"No module named numpy"
→ Run:  pip3 install numpy scipy sympy requests z3-solver
