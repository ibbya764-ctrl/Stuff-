# metacognition.py and curiosity_engine.py — usage

Two new modules that drop into the same directory as your existing
`pipeline.py`. No changes needed to existing files.

## 1. Install

Upload both files to `/content/` in Colab (same place as your other
`.py` files). They import from `pipeline.py` indirectly via the
pipeline instance you pass them — no new dependencies.

## 2. Metacognition — adversarial self-audit

Run after a normal pipeline run:

```python
from pipeline import setup_pipeline, ask
from obligation_store import ObligationStore
from metacognition import metacognitive_audit

# Set up pipeline as usual
setup_pipeline(llm_chat, existing_score_fn=auto_score_physics_branch_v3)

# Run on a question
result = ask("derive the MOND coefficient from CP2 closure")

# Audit the result
store = ObligationStore("/content/drive/MyDrive/obligation_store.json")
audit = metacognitive_audit(
    result,
    llm_chat_fn=llm_chat,
    obligation_store=store,
    record_low_confidence_as_obligations=True,
)
```

Three things happen:

1. **Losing branches generate questions about the winner.** This is the
   key change from the previous self-audit — the questions don't come
   from the winner reflecting on itself. Each losing branch raises
   questions from its own methodological perspective, then a critique
   pass keeps only the strong ones.

2. **Confidence is grounded in objective signals.** It looks at the
   verification verdict, falsification count, total_score, and whether
   the branch's assumptions match persistent unresolved gaps. The LLM
   cannot self-report higher confidence than the evidence supports —
   the ceiling is enforced in the answer prompt.

3. **Low-confidence findings become obligations.** If the audit produces
   LOW confidence, every finding is recorded back into the obligation
   store so future runs see them as persistent gaps.

## 3. Curiosity — self-directed exploration

After you've done some normal runs and the obligation store has
accumulated persistent gaps:

```python
from curiosity_engine import CuriosityEngine

# The pipeline must be set up first (with an obligation_store)
engine = CuriosityEngine(
    pipeline=_PIPELINE_INSTANCE,   # the global pipeline from setup_pipeline
    log_path="/content/drive/MyDrive/curiosity_log.json",
)

# Let the system pursue 3 of its own questions
runs = engine.pursue(n_iterations=3)
```

What it does:

1. Looks at the obligation store for persistent unresolved gaps.
2. Ranks them by blocking score (recurrence × source diversity × kind).
3. Picks the top one not yet pursued.
4. Generates a research question targeting that gap.
5. Runs the full pipeline on the generated question.
6. Logs whether the target was discharged and what techniques were learned.

You can run it once or repeatedly. It maintains its own log so it
won't pursue the same gap twice.

```python
# See what the system has been investigating
print(engine.summary())
# {'n_pursuits': 3, 'n_discharged': 1, 'discharge_rate': 0.33,
#  'n_techniques_from_curiosity': 4}
```

## 4. The combination

The two modules amplify each other:

- The metacognition module records audit findings as obligations.
- The curiosity engine then picks audit findings as targets to pursue.
- Pursuing a target may produce a verified branch.
- That verified branch is added to the technique library.
- Future runs retrieve those techniques as context.
- The system's epistemic frontier shrinks over time.

Recommended workflow when you have your local GPU running:

```python
# Normal user-driven run
result = ask("derive coefficient X from principle Y")
audit = metacognitive_audit(result, llm_chat, obligation_store=store)

# Periodically — say overnight or between user sessions —
# let the system explore its own frontier
engine.pursue(n_iterations=10)
```

That last loop is what turns the scaffold from a reactive tool into
something that compounds knowledge on its own.
