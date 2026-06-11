"""
from_scratch_trainer.py
=======================

The actual native architecture path.

Two models run simultaneously:
  Inference engine (qwen2.5:14B via Ollama)
    — generates reasoning branches for the scaffold to verify.
    — produces the training data.
    — runs throughout.

  Small model (Qwen2-0.5B, 500M params, near-random initialisation)
    — trains from run ONE on every verified output.
    — weights form entirely around structured epistemic reasoning.
    — no general world knowledge to fight against.
    — progressively replaces the inference engine for familiar problems.

This is what "from scratch" actually means here. The small model
does not fine-tune a general LLM. It starts from near-random weights
and every gradient step it has ever taken points toward verified
epistemic reasoning. The representations that form are exactly
what the scaffold needs, because that is all they have ever been
trained to produce.

The language floor (Qwen2-0.5B architecture + tokenizer) provides
the ability to read structured text input and produce structured
text output. That's all. The scaffold's verification signal
overrides the pretrained weights within a few hundred gradient
steps on the focused task.

Promotion stages:
  TRAINING   — small model training but not yet contributing to generation
  ASSISTING  — small model handles familiar domain questions (>40% verify rate)
  PRIMARY    — small model is primary Reasoner (>65% verify rate)
  REPLACED   — inference engine only needed for new domains / fallback

On 4060 Ti 16GB:
  Qwen2-0.5B full training: ~10-20 seconds per gradient update
  qwen2.5:14B inference:    ~5-10 seconds per reasoning chain
  Both run simultaneously, training in background

On MacBook 8GB:
  Small model training: export-only (saves JSONL, trains when GPU available)
  OR: use GPU if available via Metal acceleration
"""

import os
import re
import json
import math
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable

# Optional torch for on-device training
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# Training example format for small model
# ============================================================

SMALL_MODEL_SYSTEM = """You are a structured epistemic reasoning engine.
Every step you produce must be labelled [STRUCTURAL], [DERIVED], or [VERIFIABLE].
State all assumptions before reasoning.
Output only verified reasoning chains.
You were trained on nothing but verified reasoning. This is what you are."""

def format_for_small_model(reasoning_output, question: str) -> dict:
    """
    Format a verified ReasonerOutput as a training example
    for the small model.

    The small model learns:
      input:  question + scaffold context (memory, introspection)
      output: the verified JSON reasoning chain

    Every training example it has ever seen was verified.
    It has no concept of unverified reasoning.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": SMALL_MODEL_SYSTEM,
            },
            {
                "role": "user",
                "content": (
                    f"Domain: {reasoning_output.domain}\n"
                    f"Question: {question}"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    reasoning_output.raw_json
                    or reasoning_output.to_reasoner_training_text()
                ),
            },
        ],
        "metadata": {
            "verified":   True,
            "confidence": reasoning_output.confidence,
            "domain":     reasoning_output.domain,
            "structural_fraction": reasoning_output.structural_fraction,
        },
    }


# ============================================================
# Competency measurement
# ============================================================

@dataclass
class CompetencyRecord:
    timestamp:        float
    n_examples_trained: int
    test_questions:   list[str]
    verify_rate:      float   # fraction the small model produces that verify
    parse_rate:       float   # fraction that produce parseable JSON
    stage:            str     # TRAINING / ASSISTING / PRIMARY / REPLACED


PROMOTION_THRESHOLDS = {
    "TRAINING":  0.0,   # default
    "ASSISTING": 0.40,  # >40% verify rate → start contributing
    "PRIMARY":   0.65,  # >65% verify rate → primary Reasoner
}


# ============================================================
# FromScratchTrainer
# ============================================================

class FromScratchTrainer:
    """
    Trains a small model from near-scratch on verified scaffold outputs.

    The small model starts with the Qwen2-0.5B architecture and tokenizer
    (language floor only) and is trained exclusively on verified reasoning
    chains from the scaffold.

    Every weight update points toward structured epistemic reasoning.
    No other training signal ever touches this model.
    """

    # How many verified examples before measuring competency
    MEASURE_EVERY     = 50
    # Min examples before testing for promotion
    MIN_BEFORE_TEST   = 20

    def __init__(
        self,
        base_model_name: str   = "Qwen/Qwen2-0.5B",
        output_dir:      str   = "./scaffold_data/small_model",
        data_dir:        str   = "./scaffold_data/from_scratch_data",
        learning_rate:   float = 1e-4,     # higher than fine-tune: learning from scratch
        batch_size:      int   = 2,
        gradient_accum:  int   = 8,
        n_epochs_per_cycle: int = 2,
        verbose:         bool  = True,
    ):
        self.model_name    = base_model_name
        self.output_dir    = output_dir
        self.data_dir      = data_dir
        self.lr            = learning_rate
        self.batch_size    = batch_size
        self.grad_accum    = gradient_accum
        self.epochs        = n_epochs_per_cycle
        self.verbose       = verbose

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        self._examples:   list[dict]       = []
        self._competency: list[CompetencyRecord] = []
        self._stage:      str              = "TRAINING"
        self._cycle:      int              = 0
        self._lock:       threading.Lock   = threading.Lock()
        self._backend:    str              = self._detect_backend()

        self._load_examples()

        if self.verbose:
            print(
                f"[from_scratch] Initialised. Backend: {self._backend}. "
                f"Model: {base_model_name}. "
                f"Stage: {self._stage}. "
                f"Examples: {len(self._examples)}."
            )

    # ---- Backend detection ----

    def _detect_backend(self) -> str:
        try:
            import unsloth  # noqa
            return "unsloth"
        except ImportError:
            pass
        if HAS_TORCH:
            try:
                import transformers  # noqa
                import torch
                has_cuda = torch.cuda.is_available()
                has_mps  = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                return "transformers" if (has_cuda or has_mps) else "transformers_cpu"
            except ImportError:
                pass
        return "export_only"

    # ---- Accumulating training data ----

    def add_verified_run(
        self,
        reasoning_output,
        question: str,
    ) -> bool:
        """
        Add one verified run to the training corpus.
        Returns True if a training cycle was triggered.
        """
        if not reasoning_output.verified:
            return False

        example = format_for_small_model(reasoning_output, question)

        with self._lock:
            self._examples.append(example)
            n = len(self._examples)

        self._save_examples()

        if self.verbose:
            print(
                f"  [from_scratch] Example {n} added "
                f"(domain: {reasoning_output.domain}, "
                f"confidence: {reasoning_output.confidence})"
            )

        # Trigger training cycle when enough examples accumulated
        if n >= self.MIN_BEFORE_TEST and n % self.MEASURE_EVERY == 0:
            self._trigger_training_cycle()
            return True

        return False

    # ---- Training cycle ----

    def _trigger_training_cycle(self) -> None:
        """Run one training cycle on all accumulated examples."""
        if self._backend == "export_only":
            self._export_for_external_training()
            return

        # Run training in background thread so pipeline isn't blocked
        thread = threading.Thread(
            target=self._run_training_cycle,
            daemon=True,
        )
        thread.start()

    def _run_training_cycle(self) -> None:
        self._cycle += 1
        n = len(self._examples)

        if self.verbose:
            print(
                f"\n[from_scratch] Training cycle {self._cycle} "
                f"({n} examples, stage: {self._stage})..."
            )

        dataset_path = os.path.join(
            self.data_dir, f"cycle_{self._cycle}.jsonl"
        )
        self._write_dataset(dataset_path)

        if self._backend == "unsloth":
            self._train_unsloth(dataset_path)
        elif self._backend in ("transformers", "transformers_cpu"):
            self._train_transformers(dataset_path)

        if self.verbose:
            print(f"[from_scratch] Cycle {self._cycle} complete.")

    def _train_unsloth(self, dataset_path: str) -> None:
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from datasets import load_dataset

            model_source = (
                os.path.join(self.output_dir, "model")
                if os.path.exists(os.path.join(self.output_dir, "model"))
                else self.model_name
            )

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_source,
                max_seq_length=1024,
                load_in_4bit=True,
            )

            # For truly from-scratch: reinitialise weights if first cycle
            if self._cycle == 1:
                if self.verbose:
                    print(
                        "  [from_scratch] Reinitialising non-embedding weights "
                        "for near-scratch training..."
                    )
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        # Keep embedding weights (language floor)
                        # Reinitialise all other weights
                        if "embed" not in name and "lm_head" not in name:
                            torch.nn.init.normal_(param, mean=0.0, std=0.02)

            # No LoRA — train all weights from scratch
            # (the model is small enough for full training on 4060 Ti)
            for param in model.parameters():
                param.requires_grad = True

            dataset = load_dataset("json", data_files=dataset_path)["train"]

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="messages",
                max_seq_length=1024,
                args=TrainingArguments(
                    per_device_train_batch_size=self.batch_size,
                    gradient_accumulation_steps=self.grad_accum,
                    num_train_epochs=self.epochs,
                    learning_rate=self.lr,
                    fp16=True,
                    output_dir=self.output_dir,
                    save_strategy="epoch",
                    logging_steps=5,
                    optim="adamw_8bit",
                    lr_scheduler_type="cosine",
                    warmup_ratio=0.1,
                    report_to="none",
                ),
            )
            trainer.train()

            save_path = os.path.join(self.output_dir, "model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            if self.verbose:
                print(f"  [from_scratch] Model saved to {save_path}")

        except Exception as e:
            if self.verbose:
                print(f"  [from_scratch] Training error: {e}")

    def _train_transformers(self, dataset_path: str) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer,
                TrainingArguments, Trainer,
                DataCollatorForLanguageModeling,
            )
            from datasets import load_dataset

            model_source = (
                os.path.join(self.output_dir, "model")
                if os.path.exists(os.path.join(self.output_dir, "model"))
                else self.model_name
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            if self._cycle == 1:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if "embed" not in name and "lm_head" not in name:
                            torch.nn.init.normal_(param, mean=0.0, std=0.02)

            dataset = load_dataset("json", data_files=dataset_path)["train"]

            def tokenize(ex):
                msgs  = ex.get("messages", [])
                texts = " ".join(m.get("content", "") for m in msgs)
                tok   = tokenizer(
                    texts,
                    truncation=True,
                    max_length=1024,
                    padding="max_length",
                )
                tok["labels"] = tok["input_ids"].copy()
                return tok

            tokenized = dataset.map(tokenize)

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=self.output_dir,
                    num_train_epochs=self.epochs,
                    per_device_train_batch_size=self.batch_size,
                    gradient_accumulation_steps=self.grad_accum,
                    learning_rate=self.lr,
                    fp16=True,
                    save_strategy="epoch",
                    logging_steps=5,
                    report_to="none",
                    lr_scheduler_type="cosine",
                    warmup_ratio=0.1,
                ),
                train_dataset=tokenized,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            trainer.train()

            save_path = os.path.join(self.output_dir, "model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        except Exception as e:
            if self.verbose:
                print(f"  [from_scratch] Training error: {e}")

    # ---- Competency testing ----

    def test_competency(
        self,
        test_questions: list[str],
        test_fn:        Callable,   # function that runs a question through small model
        domain:         str = "physics",
    ) -> CompetencyRecord:
        """
        Test whether the small model can generate parseable, verifiable
        reasoning chains. Called periodically to track progression.

        test_fn should run a question through the small model's Reasoner
        and return a ReasonerOutput.
        """
        if not test_questions:
            return None

        n_verified = 0
        n_parseable = 0

        for q in test_questions[:5]:
            try:
                output = test_fn(q, domain)
                if output and output.steps:
                    n_parseable += 1
                    if output.verified:
                        n_verified += 1
            except Exception:
                pass

        n = max(1, len(test_questions[:5]))
        verify_rate = n_verified / n
        parse_rate  = n_parseable / n

        # Determine stage
        if verify_rate >= PROMOTION_THRESHOLDS["PRIMARY"]:
            new_stage = "PRIMARY"
        elif verify_rate >= PROMOTION_THRESHOLDS["ASSISTING"]:
            new_stage = "ASSISTING"
        else:
            new_stage = "TRAINING"

        if new_stage != self._stage:
            print(
                f"\n[from_scratch] STAGE CHANGE: "
                f"{self._stage} → {new_stage} "
                f"(verify rate: {verify_rate:.0%})"
            )
            self._stage = new_stage

        record = CompetencyRecord(
            timestamp=time.time(),
            n_examples_trained=len(self._examples),
            test_questions=test_questions[:5],
            verify_rate=round(verify_rate, 3),
            parse_rate=round(parse_rate, 3),
            stage=self._stage,
        )
        self._competency.append(vars(record))
        self._save_competency()

        if self.verbose:
            print(
                f"[from_scratch] Competency test: "
                f"verify={verify_rate:.0%} parse={parse_rate:.0%} "
                f"stage={self._stage}"
            )

        return record

    def make_reasoner_fn(self) -> Optional[Callable]:
        """
        Return an llm_chat-compatible function using the small model,
        or None if the model isn't ready or loaded.
        Only returns a function if stage is ASSISTING or PRIMARY.
        """
        if self._stage == "TRAINING":
            return None

        model_path = os.path.join(self.output_dir, "model")
        if not os.path.exists(model_path):
            return None

        if self._backend == "export_only":
            return None

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model     = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto"
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1000,
                do_sample=False,
            )

            def small_model_chat(system: str, user: str) -> str:
                prompt = f"{system}\n\nUser: {user}\n\nAssistant:"
                out    = pipe(prompt)[0]["generated_text"]
                return out[len(prompt):].strip()

            return small_model_chat

        except Exception as e:
            if self.verbose:
                print(f"[from_scratch] Could not load small model: {e}")
            return None

    # ---- Export for external training (MacBook / export_only) ----

    def _export_for_external_training(self) -> None:
        path = os.path.join(self.data_dir, "all_examples.jsonl")
        self._write_dataset(path)
        if self.verbose:
            print(
                f"  [from_scratch] Exported {len(self._examples)} examples "
                f"to {path}"
            )
            print(
                f"  Install unsloth on 4060 Ti to enable live training:\n"
                f"    pip install unsloth"
            )

    def _write_dataset(self, path: str) -> None:
        with self._lock:
            examples = list(self._examples)
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    # ---- Status ----

    def status(self) -> str:
        lines = [
            "FromScratchTrainer:",
            f"  Model:        {self.model_name} (500M params)",
            f"  Backend:      {self._backend}",
            f"  Stage:        {self._stage}",
            f"  Training data: {len(self._examples)} verified examples",
            f"  Cycles:       {self._cycle}",
        ]
        if self._competency:
            last = self._competency[-1]
            lines.append(
                f"  Last test:    verify={last['verify_rate']:.0%} "
                f"parse={last['parse_rate']:.0%}"
            )
        if self._stage == "TRAINING":
            needed = max(0, self.MIN_BEFORE_TEST - len(self._examples))
            if needed > 0:
                lines.append(
                    f"  First test in {needed} more verified examples"
                )
            lines.append(
                f"  Promotes to ASSISTING at {PROMOTION_THRESHOLDS['ASSISTING']:.0%} verify rate"
            )
        elif self._stage == "ASSISTING":
            lines.append(
                f"  Promotes to PRIMARY at "
                f"{PROMOTION_THRESHOLDS['PRIMARY']:.0%} verify rate"
            )
        elif self._stage == "PRIMARY":
            lines.append(
                "  Small model is primary Reasoner for familiar domains."
            )
        return "\n".join(lines)

    # ---- Persistence ----

    def _load_examples(self) -> None:
        path = os.path.join(self.data_dir, "all_examples.jsonl")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                self._examples = [json.loads(l) for l in f if l.strip()]
        except (json.JSONDecodeError, OSError):
            pass

    def _save_examples(self) -> None:
        path = os.path.join(self.data_dir, "all_examples.jsonl")
        self._write_dataset(path)

    def _save_competency(self) -> None:
        path = os.path.join(self.output_dir, "competency.json")
        with open(path, "w") as f:
            json.dump({"records": self._competency}, f, indent=2)
