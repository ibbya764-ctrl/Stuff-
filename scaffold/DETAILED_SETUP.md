# Full Setup Guide
# Python 3.12.5 + Ollama on Mac

---

## PART 1 — Verify Python

You already have Python 3.12.5. Let's confirm it's working.

Open Terminal (press Cmd+Space, type Terminal, press Enter).

Type this and press Enter:
    python3 --version

You should see:
    Python 3.12.5

If it says a different version (like 3.14), type this instead:
    python3.12 --version

Use whichever command shows 3.12.5 — that's what you'll use
for everything in this guide.

---

## PART 2 — Install pip packages

In Terminal, run this one command:

    pip3 install numpy scipy sympy requests z3-solver

If that gives a "not found" error, try:

    python3.12 -m pip install numpy scipy sympy requests z3-solver

You'll see it download and install each package. Takes about
1-2 minutes. You'll know it's done when the prompt comes back.

---

## PART 3 — Install Ollama

Ollama is what runs the AI model locally on your Mac.

1. Go to:  https://ollama.com

2. Click the big "Download" button (it detects Mac automatically).

3. Open the downloaded file — you'll see a window asking you to
   drag Ollama into your Applications folder. Do that.

4. Open Ollama from your Applications folder.
   You'll see a small llama icon appear in your menu bar (top right
   of your screen, near the clock). That means it's running.

---

## PART 4 — Download the AI model

With Ollama running, open a Terminal window and run:

    ollama pull qwen2.5:7b

This downloads the AI model — about 4.5 GB.
It will show a progress bar. Takes 5-15 minutes depending
on your internet speed.

You'll know it's done when you see something like:
    success

---

## PART 5 — Set up your scaffold folder

1. Download scaffold_complete.zip from Claude (it's in the files
   above this message).

2. In Terminal, run:

    cd ~/Desktop
    unzip ~/Downloads/scaffold_complete.zip -d scaffold

   (This assumes it downloaded to your Downloads folder.
    If it's somewhere else, adjust the path.)

3. Go into the folder:

    cd ~/Desktop/scaffold

4. Check the files are there:

    ls

   You should see a long list of .py files.

---

## PART 6 — Run the system

You need TWO Terminal windows open at the same time.

### Window 1 — Start the Ollama server

Open a new Terminal window (Cmd+T in Terminal, or Cmd+N).
Run:

    ollama serve

You'll see something like:
    Listening on 127.0.0.1:11434

LEAVE THIS WINDOW OPEN. Don't close it or press Ctrl+C.
It needs to keep running in the background.

### Window 2 — Run the scaffold

Open another new Terminal window.
Run:

    cd ~/Desktop/scaffold
    python3 mac_run.py

The first time you run this you'll see it:
- Check all the modules are loading (should all show "ok")
- Initialise the enhanced pipeline (takes 5-10 seconds)
- Run a first physics question automatically

This first question will take 30-90 seconds because the model
is generating a response. On an M-chip Mac it'll be around
30 seconds. On an older Intel Mac, up to 90 seconds.
That's normal.

---

## PART 7 — Using it

After the first question runs, you'll see a prompt:

    >

Type any question and press Enter. For example:
    > What is the relationship between MOND and dark matter?
    > Derive the Euler-Lagrange equations from first principles
    > How does mean field theory work?

Special commands:
    explore      — the system picks its own question to explore
    consolidate  — runs maintenance (merge duplicates, tune params)
    status       — shows training buffer and learning progress
    Ctrl+C       — stops the session and saves training data

Everything gets saved to a folder called scaffold_data/
automatically. Next time you run mac_run.py it picks up
where it left off.

---

## If something goes wrong

### "No module named X"
The file for that module isn't in the folder. Check that
all .py files from the zip are in ~/Desktop/scaffold/

### "Connection refused" or LLM errors
The Ollama server isn't running. Make sure Window 1 has
ollama serve running and you haven't closed it.

### Responses are very slow (Intel Mac)
Normal — Intel Macs run the model on CPU only. Each response
takes 60-120 seconds. Everything still works, just slower.
Consider using the Claude API instead for speed:
    export ANTHROPIC_API_KEY=your_key_here
    python3 mac_run.py
(Get a key at https://console.anthropic.com)

### "python3: command not found"
Try python3.12 instead of python3 in all the commands above.

### The zip doesn't extract properly
Try double-clicking the zip file in Finder instead of
using the Terminal command. Then move all the files into
a folder on your Desktop called "scaffold".

---

## What's happening when it runs

When you ask a question, the system:
1. Checks if this is a new domain (and bootstraps it if so)
2. Searches its memory for similar past questions
3. Generates 3 branches (reasoning approaches)
4. Verifies each branch with symbolic math (sympy)
5. Picks the best verified branch
6. Checks how novel the result is
7. Saves the run to its episodic memory
8. Updates the training buffer

After every 5 verified runs, it packages up the training
data and saves it to scaffold_data/training_data.jsonl —
ready for fine-tuning on the 4060 Ti when it arrives.

---

## Tomorrow — switching to the 4060 Ti

When the GPU arrives, install unsloth on the new machine:
    pip install unsloth

Then run mac_run.py exactly the same way. The trainer
auto-detects the GPU and switches from "save dataset"
mode to live LoRA fine-tuning. The training data
accumulated on the Mac transfers over automatically
(just copy the scaffold_data/ folder).
