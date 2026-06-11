# local_model.py + typed_output.py — usage

The foundation layer for running the scaffold on local hardware. Two
files, working together to make small local models behave reliably.

## What this solves

The pipeline expects `llm_chat_fn(system, user) -> str`. Up to now,
that's been an API call. On the RTX 5070 it can be a local model
running at zero marginal cost, but local models have one big problem
the API didn't: **they hallucinate output format**. A 7B model asked
for JSON will sometimes return YAML, sometimes wrap it in three
fenced blocks, sometimes drop a closing brace, sometimes invent extra
keys, sometimes write "None" instead of "null". Without a layer that
handles all of that, the pipeline breaks constantly.

These two files solve both halves:

- **`local_model.py`**: four backends (stub, llama.cpp, vLLM, hosted
  API) plus a hybrid router, all conforming to the same
  `chat(system, user) -> str` interface that the pipeline already
  uses. Drop-in replacement. Built-in telemetry — every call is timed
  and logged.

- **`typed_output.py`**: defensive JSON extraction (handles fences,
  prose, trailing commas, python-isms, single quotes), schema
  validation, and a repair-by-retry loop. When the model returns
  malformed JSON, the layer feeds the error back to the model and
  asks it to fix the specific problem.

## Before the GPU arrives — testing the pipeline

Use the stub backend to exercise the rest of the system without
needing a real model:

```python
from local_model import setup_stub
from pipeline import setup_pipeline

chat, telemetry = setup_stub()
setup_pipeline(chat, existing_score_fn=auto_score_physics_branch_v3)

# Now run the pipeline as normal — calls will return canned JSON
# that's structurally valid, so you can test that downstream parsing,
# obligation store updates, and module integration all work without
# spending API credits.
```

The stub returns deterministic JSON for branch generation, audit
questions, and graph extraction — the three commonest call types.

## After the GPU arrives — local-first

Once you have the RTX 5070 set up and a GGUF model downloaded
(start with Qwen2.5-7B-Instruct at Q4_K_M, about 4.4 GB):

```python
from local_model import setup_local_qwen
from pipeline import setup_pipeline

chat, telemetry = setup_local_qwen(
    model_path="/path/to/qwen2.5-7b-instruct-q4_k_m.gguf",
)
setup_pipeline(chat, existing_score_fn=auto_score_physics_branch_v3)
```

That's it. Everything else in the pipeline works unchanged.

## Hybrid mode — when local isn't enough

For hard problems, hybrid mode keeps most calls local and only sends
the difficult ones to a hosted API:

```python
from local_model import setup_hybrid

chat, telemetry = setup_hybrid(
    local_model_path="/path/to/qwen2.5-7b-instruct-q4_k_m.gguf",
    hosted_provider="anthropic",
    hosted_model="claude-sonnet-4-6",
)
```

Routing rules live in `DEFAULT_HOSTED_TAGS`:
- `branch_gen_hard`     → hosted
- `structure_extract`   → hosted (SME extraction is fragile)
- `metacog_audit_hard`  → hosted
- everything else       → local

You can override per-call with `force_backend="local"` or
`force_backend="hosted"`. Tags are passed through to telemetry.

## Using typed_call

The pipeline currently parses JSON loosely with regex. Once you switch
to a local model, use `typed_call` for any call that needs structured
output:

```python
from typed_output import typed_call, BRANCHES_SCHEMA

result = typed_call(
    chat,
    system=BRANCH_GEN_SYSTEM,
    user=f"Question: {question}",
    schema=BRANCHES_SCHEMA,
    max_retries=2,
    tag="branch_gen",
)

if result.valid:
    branches = result.value["branches"]
else:
    print(f"Failed after {result.attempts} attempts: {result.errors}")
    branches = []   # graceful fallback
```

Pre-built schemas for the main pipeline call types:
- `BRANCHES_SCHEMA`      — branch generation
- `AUDIT_QUESTIONS_SCHEMA` — metacognitive audit questions
- `CRITIQUE_SCHEMA`      — question quality critique
- `GRAPH_EXTRACTION_SCHEMA` — Structure Mapper entity/relation extraction
- `CURIOSITY_QUESTION_SCHEMA` — exploratory question generation

## Verified behaviour

Smoke tests cover:
- ✓ Stub backend produces structurally valid responses for all
  pipeline call types
- ✓ JSON extraction handles fences, prose, trailing commas, Python
  `None`/`True`, single quotes, nested structures
- ✓ Schema validation catches missing keys, wrong types, length
  violations, and non-object inputs
- ✓ Retry loop recovers from a deliberately-flakey backend that
  returns garbage twice before producing valid output
- ✓ Telemetry logs every call with timing, success status, and
  caller-supplied tags

## Hardware notes for the RTX 5070

12 GB VRAM. Use Q4_K_M GGUF as the default. Practical models:

- **Qwen2.5-7B-Instruct Q4_K_M**: ~4.4 GB, 50-100 tok/s, great
  instruction following → recommended starting point
- **Qwen3-8B (hybrid thinking)**: ~5 GB, supports `mode="thinking"`
  for hard problems, slower → use for audits/critiques
- **DeepSeek-R1-Distill-Qwen-7B Q4_K_M**: ~4.4 GB, strong on
  step-by-step reasoning → use for branch generation
- **Qwen2.5-14B Q4_K_M**: ~8.5 GB, slower (~25-40 tok/s) but
  noticeably better → use when 7B isn't enough

NOT recommended for the 5070:
- Anything above 14B at Q4 (OOM or thrashing)
- Long context (>16k) with the 14B (KV cache eats VRAM)
- Local fine-tuning (12 GB is too tight for serious LoRA)

## What this enables next

With local model running:
1. The **curiosity engine** can run overnight without burning API
   credits. This is the biggest single unlock — autonomous self-
   directed exploration on the obligation store.
2. The **evaluation harness** (next module to build) can run on every
   change to measure whether modifications are helping.
3. The **verifier stack improvements** (Z3, property-based testing)
   become worth investing in — local-only calls let you verify
   aggressively.

What this doesn't yet do (planned):
- Streaming output — currently returns complete strings only.
- Batching multiple calls in one forward pass — vLLM supports this
  but the wrapper doesn't expose it yet.
- Speculative decoding with a draft model — possible on the 5070 with
  Qwen2.5-1.5B as draft + Qwen2.5-7B as target.

## File locations

- `local_model.py` — the backend layer
- `typed_output.py` — JSON extraction, validation, retry
- `local_model_telemetry.json` — call log, auto-created at runtime
