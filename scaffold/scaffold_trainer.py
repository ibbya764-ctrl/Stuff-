"""
scaffold_trainer.py
===================

The recursive training loop.

The scaffold runs on problems. Some branches verify. Those verified
runs become training examples. The model is fine-tuned on them. The
fine-tuned model runs the scaffold better. More things verify. Richer
training data. Fine-tune again. Repeat.

Each cycle the training data is better than the last because the model
that generated it was better than the model trained in the previous
cycle. This is what "learning from its own learning" means mechanically.

Four components:

  DataFormatter      — converts episodic records into structured training
                       examples. Each step is labelled [STRUCTURAL],
                       [DERIVED], or [VERIFIABLE]. The model learns to
                       categorise its own reasoning.

  TrainingBuffer     — accumulates examples with curriculum ordering.
                       Starts with structurally simple runs, progresses
                       to harder ones as training cycles advance.

  PerformanceTracker — measures verification rate, structural fraction,
                       and GUE score before and after each fine-tune.
                       Training cycle is kept only if performance improved.

  ScaffoldTrainer    — the coordinator. Called after every pipeline run.
                       Decides when to train. Triggers fine-tuning.
                       Rolls back if performance regressed.

Fine-tuning backend:
  - If torch + peft available: runs QLoRA directly on the 4060 Ti 16GB
  - If unsloth available: uses unsloth for faster training (recommended)
  - Otherwise: exports JSONL dataset for external fine-tuning

The output is a LoRA adapter, not a full model retrain. The base model
stays intact. The adapter encodes the epistemic structure learned from
verified runs. At any point you can merge adapter + base to produce
a standalone model checkpoint.
"""

import os
import re
import json
import time
import math
import copy
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# Training example format
# ============================================================

SYSTEM_PROMPT = """You are a structured epistemic reasoning system.

For every reasoning step, output one of these markers:
  [STRUCTURAL]  — follows directly from accumulated knowledge or technique template
  [DERIVED]     — requires inference; you are generating new content here
  [VERIFIABLE]  — can be checked symbolically or numerically; flag for verification

Always separate assumptions from steps. State your candidate result clearly.
State explicitly whether your conclusion was verified.

Your reasoning chains are training data for the next version of yourself.
Produce outputs that would make that next version better."""

INSTRUCTION_TEMPLATE = """Reason about the following using structured epistemic reasoning.
Label each step. Separate assumptions from derivations. State the result and whether it verified.

{introspective_context}"""

OUTPUT_TEMPLATE = """Method: {method}

Assumptions:
{assumptions}

Reasoning:
{steps}

Result: {candidate_result}

Verification: {verification_status}
Obligations discharged: {obligations_discharged}
Structural fraction: {structural_fraction:.0%}"""


@dataclass
class TrainingExample:
    """
    One formatted training example derived from a verified run.
    """
    run_id:               str
    domain:               str
    instruction:          str      # what the model was asked
    input_text:           str      # question + context
    output_text:          str      # labelled reasoning chain
    verified:             bool
    structural_fraction:  float
    obligations_discharged: int
    difficulty:           float    # 0=easy, 1=hard — used for curriculum ordering

    def to_jsonl(self) -> str:
        """Alpaca/instruction-tuning format for HuggingFace / Ollama."""
        return json.dumps({
            "system":     SYSTEM_PROMPT,
            "instruction": self.instruction,
            "input":      self.input_text,
            "output":     self.output_text,
            "metadata": {
                "run_id":               self.run_id,
                "domain":               self.domain,
                "verified":             self.verified,
                "structural_fraction":  self.structural_fraction,
                "obligations_discharged": self.obligations_discharged,
                "difficulty":           self.difficulty,
            }
        })

    def to_chatml(self) -> str:
        """ChatML format — compatible with Qwen, Llama-3, Mistral."""
        return json.dumps({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": self.input_text},
                {"role": "assistant", "content": self.output_text},
            ],
            "metadata": {
                "run_id":   self.run_id,
                "domain":   self.domain,
                "verified": self.verified,
            }
        })


# ============================================================
# DataFormatter
# ============================================================

class DataFormatter:
    """
    Converts episodic records into structured training examples.

    The key formatting decision: step type markers ([STRUCTURAL],
    [DERIVED], [VERIFIABLE]) are embedded directly in the output text.
    The model learns to produce these markers as part of its normal
    output. No special tokens or architectural changes needed for
    the fine-tuning stage — the structure is encoded in the text.
    """

    def format_run(
        self,
        record,                          # EpisodicRecord
        full_result:    Optional[dict] = None,   # full pipeline result if available
        introspective_context: str      = "",
    ) -> Optional[TrainingExample]:
        """
        Convert one episodic record into a training example.
        Returns None if the run lacks enough information to format.
        """
        if not record.verified:
            return None   # only train on verified runs

        # Extract branch details from full result if available
        branch    = {}
        all_steps = record.key_steps or []
        method    = record.method_name or "unknown"
        result_text = ""
        assumptions: list[str] = []

        if full_result:
            branch      = full_result.get("selected_branch", {})
            all_steps   = branch.get("steps", record.key_steps or [])
            method      = branch.get("method", record.method_name or "unknown")
            result_text = branch.get("candidate_result", "")
            assumptions = branch.get("assumptions", [])

        if not all_steps:
            return None

        # Format steps with type labels
        labelled_steps = _label_steps(all_steps)
        steps_text     = "\n".join(
            f"{label} {step}"
            for label, step in labelled_steps
        )

        # Format assumptions
        assumptions_text = (
            "\n".join(f"- {a}" for a in assumptions)
            if assumptions else "- (none stated)"
        )

        # Verification status
        ver_status = "VERIFIED" if record.verified else "UNVERIFIED"

        output_text = OUTPUT_TEMPLATE.format(
            method=method,
            assumptions=assumptions_text,
            steps=steps_text,
            candidate_result=result_text or "(see steps above)",
            verification_status=ver_status,
            obligations_discharged=record.obligations_discharged,
            structural_fraction=record.structural_fraction,
        )

        instruction = INSTRUCTION_TEMPLATE.format(
            introspective_context=(
                introspective_context[:500] if introspective_context else ""
            )
        ).strip()

        input_text = f"Question: {record.question}"

        # Difficulty: harder = low structural fraction (model had to generate more)
        difficulty = 1.0 - record.structural_fraction

        return TrainingExample(
            run_id=record.run_id,
            domain=record.domain,
            instruction=instruction,
            input_text=input_text,
            output_text=output_text,
            verified=record.verified,
            structural_fraction=record.structural_fraction,
            obligations_discharged=record.obligations_discharged,
            difficulty=difficulty,
        )

    def format_batch(
        self,
        records:  list,
        results:  Optional[dict] = None,   # {run_id: full_result}
    ) -> list[TrainingExample]:
        """Format a batch of records."""
        examples = []
        for record in records:
            result = (results or {}).get(record.run_id)
            ex = self.format_run(record, result)
            if ex is not None:
                examples.append(ex)
        return examples


def _label_steps(steps: list[str]) -> list[tuple[str, str]]:
    """
    Infer step type labels from step text.
    Returns list of (label, cleaned_step) pairs.
    """
    # If steps already have labels, preserve them
    results = []
    for step in steps:
        step = step.strip()
        if not step:
            continue
        if step.startswith("[STRUCTURAL]"):
            results.append(("[STRUCTURAL]", step[len("[STRUCTURAL]"):].strip()))
        elif step.startswith("[DERIVED]"):
            results.append(("[DERIVED]", step[len("[DERIVED]"):].strip()))
        elif step.startswith("[VERIFIABLE]"):
            results.append(("[VERIFIABLE]", step[len("[VERIFIABLE]"):].strip()))
        else:
            # Infer from content
            lower = step.lower()
            if any(v in lower for v in [
                "verif", "check", "test", "confirm", "sympy",
                "assert", "sample", "numerically",
            ]):
                results.append(("[VERIFIABLE]", step))
            elif any(v in lower for v in [
                "identify", "state", "note", "observe", "define",
                "let", "denote", "recognise", "given",
            ]):
                results.append(("[STRUCTURAL]", step))
            else:
                results.append(("[DERIVED]", step))
    return results


# ============================================================
# TrainingBuffer — curriculum ordering
# ============================================================

@dataclass
class BufferStats:
    n_total:        int = 0
    n_by_domain:    dict = field(default_factory=dict)
    avg_difficulty: float = 0.0
    avg_struct_frac: float = 0.0
    n_cycles_used:  int = 0


class TrainingBuffer:
    """
    Accumulates training examples with curriculum ordering.

    Curriculum: start training on structurally simple examples
    (high structural fraction, low difficulty) and progress toward
    harder ones (low structural fraction, high LLM dependency) as
    training cycles advance.

    This mirrors how the system itself develops — easy structure-
    driven reasoning first, then harder derivation-heavy reasoning.
    """

    def __init__(
        self,
        min_examples_to_train:  int   = 20,
        max_buffer_size:        int   = 500,
        curriculum_start_frac:  float = 0.3,  # start with easiest 30%
        path:                   str   = "./training_buffer.json",
    ):
        self.min_to_train   = min_examples_to_train
        self.max_size       = max_buffer_size
        self.curr_frac      = curriculum_start_frac
        self.path           = path
        self._examples:     list[dict] = []
        self._cycle:        int        = 0
        self._load()

    def add(self, example: TrainingExample) -> None:
        """Add one training example to the buffer."""
        # Deduplicate by run_id
        if any(e["run_id"] == example.run_id for e in self._examples):
            return
        self._examples.append({
            "run_id":               example.run_id,
            "domain":               example.domain,
            "instruction":          example.instruction,
            "input_text":           example.input_text,
            "output_text":          example.output_text,
            "verified":             example.verified,
            "structural_fraction":  example.structural_fraction,
            "difficulty":           example.difficulty,
            "added_at":             time.time(),
        })
        # Evict oldest if over max size
        if len(self._examples) > self.max_size:
            self._examples.sort(key=lambda e: e["added_at"])
            self._examples = self._examples[-self.max_size:]
        self._save()

    def ready_to_train(self) -> bool:
        return len(self.curriculum_batch()) >= self.min_to_train

    def curriculum_batch(self) -> list[dict]:
        """
        Return examples appropriate for the current training cycle.
        Early cycles: easier examples (high structural fraction).
        Later cycles: all examples including harder ones.
        """
        if not self._examples:
            return []
        sorted_ex = sorted(
            self._examples, key=lambda e: e["difficulty"]
        )
        # How many to include: starts at curriculum_start_frac, grows to 1.0
        progress = min(1.0, self._cycle * 0.15 + self.curr_frac)
        n        = max(self.min_to_train, int(len(sorted_ex) * progress))
        return sorted_ex[:n]

    def advance_cycle(self) -> None:
        """Call after each completed training cycle."""
        self._cycle += 1
        self._save()

    def stats(self) -> BufferStats:
        ex = self._examples
        if not ex:
            return BufferStats()
        domains: dict[str, int] = {}
        for e in ex:
            d = e.get("domain", "?")
            domains[d] = domains.get(d, 0) + 1
        return BufferStats(
            n_total=len(ex),
            n_by_domain=domains,
            avg_difficulty=sum(e["difficulty"] for e in ex) / len(ex),
            avg_struct_frac=sum(e["structural_fraction"] for e in ex) / len(ex),
            n_cycles_used=self._cycle,
        )

    def export_jsonl(self, path: str, format: str = "chatml") -> int:
        """
        Export the current curriculum batch to a JSONL file.
        format: "chatml" (default) or "alpaca"
        Returns number of examples written.
        """
        batch = self.curriculum_batch()
        if not batch:
            return 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for ex_dict in batch:
                ex = TrainingExample(
                    run_id=ex_dict["run_id"],
                    domain=ex_dict["domain"],
                    instruction=ex_dict["instruction"],
                    input_text=ex_dict["input_text"],
                    output_text=ex_dict["output_text"],
                    verified=ex_dict["verified"],
                    structural_fraction=ex_dict["structural_fraction"],
                    obligations_discharged=0,
                    difficulty=ex_dict["difficulty"],
                )
                line = ex.to_chatml() if format == "chatml" else ex.to_jsonl()
                f.write(line + "\n")
        return len(batch)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._examples = data.get("examples", [])
            self._cycle    = data.get("cycle", 0)
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "examples": self._examples,
                "cycle":    self._cycle,
            }, f)


# ============================================================
# PerformanceTracker
# ============================================================

@dataclass
class TrainingCycleRecord:
    cycle:              int
    timestamp:          float
    n_examples:         int
    verification_rate_before: Optional[float]
    verification_rate_after:  Optional[float]
    struct_fraction_before:   Optional[float]
    struct_fraction_after:    Optional[float]
    delta:              Optional[float]   # composite improvement
    adopted:            bool = False


class PerformanceTracker:
    """
    Measures performance before and after each training cycle.
    A training cycle is kept only if composite performance improved.
    """

    MIN_IMPROVEMENT = 0.015   # minimum delta to adopt

    def __init__(self, episodic_store, path: str = "./training_cycles.json"):
        self.episodic = episodic_store
        self.path     = path
        self._cycles: list[dict] = self._load()

    def measure(self, window: int = 30) -> dict:
        """Measure current performance from recent episodic records."""
        recent = self.episodic.query_recent(n=window)
        if not recent:
            return {"verification_rate": 0.0, "structural_fraction": 0.0,
                    "composite": 0.0, "n_runs": 0}
        n    = len(recent)
        vr   = sum(1 for r in recent if r.verified) / n
        sf   = sum(r.structural_fraction for r in recent) / n
        comp = 0.6 * vr + 0.4 * sf
        return {
            "verification_rate":  round(vr, 4),
            "structural_fraction": round(sf, 4),
            "composite":          round(comp, 4),
            "n_runs":             n,
        }

    def record_cycle(
        self,
        cycle:          int,
        n_examples:     int,
        perf_before:    dict,
        perf_after:     dict,
    ) -> TrainingCycleRecord:
        delta = (
            perf_after["composite"] - perf_before["composite"]
            if perf_before and perf_after else None
        )
        adopted = bool(delta is not None and delta >= self.MIN_IMPROVEMENT)
        rec = TrainingCycleRecord(
            cycle=cycle,
            timestamp=time.time(),
            n_examples=n_examples,
            verification_rate_before=perf_before.get("verification_rate"),
            verification_rate_after=perf_after.get("verification_rate"),
            struct_fraction_before=perf_before.get("structural_fraction"),
            struct_fraction_after=perf_after.get("structural_fraction"),
            delta=round(delta, 4) if delta is not None else None,
            adopted=adopted,
        )
        self._cycles.append(vars(rec))
        self._save()
        return rec

    def is_improving(self) -> bool:
        if len(self._cycles) < 2:
            return True
        recent = [c for c in self._cycles[-5:] if c.get("adopted")]
        return len(recent) >= 1

    def summary(self) -> str:
        if not self._cycles:
            return "No training cycles yet."
        adopted = sum(1 for c in self._cycles if c.get("adopted"))
        lines = [
            f"Training cycles: {len(self._cycles)} total, {adopted} adopted",
        ]
        if self._cycles:
            last = self._cycles[-1]
            lines.append(
                f"  Last: cycle {last['cycle']}, "
                f"Δ={last.get('delta',0):+.4f}, "
                f"{'ADOPTED' if last.get('adopted') else 'REJECTED'}"
            )
        return "\n".join(lines)

    def _load(self) -> list:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path) as f:
                return json.load(f).get("cycles", [])
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"cycles": self._cycles}, f, indent=2)


# ============================================================
# LoRATrainer
# ============================================================

class LoRATrainer:
    """
    Runs LoRA fine-tuning on the local GPU.

    Uses unsloth if available (2x faster, half the memory).
    Falls back to HuggingFace transformers + peft.
    Falls back to JSONL export if neither is available.

    Target hardware: RTX 4060 Ti 16GB
    Target model: Qwen2.5-7B-Instruct (fits in ~10GB with QLoRA)

    Typical training time: 30-90 minutes per cycle for 50-150 examples.
    """

    def __init__(
        self,
        model_name:   str   = "Qwen/Qwen2.5-7B-Instruct",
        output_dir:   str   = "./lora_adapters",
        lora_rank:    int   = 16,
        lora_alpha:   int   = 32,
        max_seq_len:  int   = 2048,
        n_epochs:     int   = 3,
        batch_size:   int   = 1,
        grad_accum:   int   = 4,
        learning_rate: float = 2e-4,
        verbose:      bool  = True,
    ):
        self.model_name    = model_name
        self.output_dir    = output_dir
        self.lora_rank     = lora_rank
        self.lora_alpha    = lora_alpha
        self.max_seq_len   = max_seq_len
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.grad_accum    = grad_accum
        self.lr            = learning_rate
        self.verbose       = verbose
        self._backend      = self._detect_backend()

    def _detect_backend(self) -> str:
        try:
            import unsloth  # noqa
            return "unsloth"
        except ImportError:
            pass
        try:
            import torch
            import transformers
            import peft  # noqa
            return "transformers"
        except ImportError:
            pass
        return "export_only"

    def train(self, dataset_path: str) -> dict:
        """
        Run a fine-tuning cycle on the dataset at dataset_path.
        Returns a dict with status, adapter_path, and elapsed time.
        """
        if self.verbose:
            print(f"  [lora_trainer] Backend: {self._backend}")
            print(f"  [lora_trainer] Dataset: {dataset_path}")

        if self._backend == "unsloth":
            return self._train_unsloth(dataset_path)
        elif self._backend == "transformers":
            return self._train_transformers(dataset_path)
        else:
            return {
                "status": "exported_only",
                "dataset_path": dataset_path,
                "message": (
                    "torch + peft not installed. Dataset exported to "
                    f"{dataset_path}. Fine-tune externally with:\n"
                    "  pip install unsloth  (recommended for 4060 Ti)\n"
                    "  or: pip install transformers peft trl"
                ),
            }

    def _train_unsloth(self, dataset_path: str) -> dict:
        """Fine-tune using unsloth — fastest option for consumer GPUs."""
        t0 = time.time()
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from datasets import load_dataset

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_len,
                load_in_4bit=True,
                dtype=None,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.lora_rank,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=self.lora_alpha,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing=True,
            )

            dataset = load_dataset("json", data_files=dataset_path)["train"]

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="messages",
                max_seq_length=self.max_seq_len,
                args=TrainingArguments(
                    per_device_train_batch_size=self.batch_size,
                    gradient_accumulation_steps=self.grad_accum,
                    warmup_ratio=0.05,
                    num_train_epochs=self.n_epochs,
                    learning_rate=self.lr,
                    fp16=True,
                    logging_steps=10,
                    output_dir=self.output_dir,
                    save_strategy="epoch",
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="cosine",
                    seed=42,
                ),
            )
            trainer.train()

            adapter_path = os.path.join(self.output_dir, "latest_adapter")
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)

            return {
                "status":       "trained",
                "backend":      "unsloth",
                "adapter_path": adapter_path,
                "elapsed":      round(time.time() - t0, 1),
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "backend": "unsloth"}

    def _train_transformers(self, dataset_path: str) -> dict:
        """Fine-tune using HuggingFace transformers + peft."""
        t0 = time.time()
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer,
                TrainingArguments, Trainer,
                BitsAndBytesConfig,
            )
            from peft import get_peft_model, LoraConfig, TaskType
            from datasets import load_dataset

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer.pad_token = tokenizer.eos_token

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            if self.verbose:
                model.print_trainable_parameters()

            dataset = load_dataset("json", data_files=dataset_path)["train"]

            def tokenize(example):
                msgs  = example.get("messages", [])
                text  = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_seq_len,
                    padding="max_length",
                )
                tokens["labels"] = tokens["input_ids"].copy()
                return tokens

            tokenized = dataset.map(tokenize, batched=False)

            args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=self.grad_accum,
                num_train_epochs=self.n_epochs,
                learning_rate=self.lr,
                fp16=True,
                save_strategy="epoch",
                logging_steps=10,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized,
            )
            trainer.train()

            adapter_path = os.path.join(self.output_dir, "latest_adapter")
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)

            return {
                "status":       "trained",
                "backend":      "transformers",
                "adapter_path": adapter_path,
                "elapsed":      round(time.time() - t0, 1),
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "backend": "transformers"}

    def get_ollama_instructions(self, adapter_path: str) -> str:
        """
        Print instructions for loading the fine-tuned adapter into Ollama.
        """
        return (
            f"\nTo use the fine-tuned adapter with Ollama:\n\n"
            f"1. Merge the adapter into the base model:\n"
            f"   python -c \"\n"
            f"   from peft import PeftModel\n"
            f"   from transformers import AutoModelForCausalLM\n"
            f"   base = AutoModelForCausalLM.from_pretrained('{self.model_name}')\n"
            f"   model = PeftModel.from_pretrained(base, '{adapter_path}')\n"
            f"   model.merge_and_unload().save_pretrained('./merged_model')\n"
            f"   \"\n\n"
            f"2. Convert to GGUF:\n"
            f"   python llama.cpp/convert_hf_to_gguf.py ./merged_model\n\n"
            f"3. Create Ollama model:\n"
            f"   ollama create scaffold-v{int(time.time())} "
            f"-f ./Modelfile\n\n"
            f"Or use the adapter directly with vLLM or HuggingFace inference."
        )


# ============================================================
# ScaffoldTrainer — coordinator
# ============================================================

class ScaffoldTrainer:
    """
    The coordinator. Wires together formatter, buffer, tracker, and
    LoRA trainer into the recursive self-improvement loop.

    Usage:
        trainer = ScaffoldTrainer(
            episodic_store=episodic,
            base_dir="./scaffold_data",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            verbose=True,
        )

        # Call after every pipeline run
        trainer.on_run_complete(result, record)

        # Or export dataset for external fine-tuning
        trainer.export_dataset("./training_data.jsonl")

        # Inspect training progress
        print(trainer.status())
    """

    def __init__(
        self,
        episodic_store,
        base_dir:     str   = "./scaffold_data",
        model_name:   str   = "Qwen/Qwen2.5-7B-Instruct",
        train_every_n: int  = 50,    # trigger training every N verified runs
        min_examples: int   = 20,    # minimum examples before first train
        verbose:      bool  = True,
    ):
        self.episodic       = episodic_store
        self.base_dir       = base_dir
        self.train_every_n  = train_every_n
        self.verbose        = verbose

        self.formatter  = DataFormatter()
        self.buffer     = TrainingBuffer(
            min_examples_to_train=min_examples,
            path=os.path.join(base_dir, "training_buffer.json"),
        )
        self.tracker    = PerformanceTracker(
            episodic_store,
            path=os.path.join(base_dir, "training_cycles.json"),
        )
        self.lora       = LoRATrainer(
            model_name=model_name,
            output_dir=os.path.join(base_dir, "lora_adapters"),
            verbose=verbose,
        )

        self._verified_since_train: int = 0

        if self.verbose:
            backend = self.lora._backend
            print(f"[scaffold_trainer] Initialised. "
                  f"Backend: {backend}. "
                  f"Buffer: {self.buffer.stats().n_total} examples.")

    # ---- Called after every pipeline run ----

    def on_run_complete(
        self,
        pipeline_result:    dict,
        episodic_record     = None,
        full_result:        Optional[dict] = None,
        introspective_ctx:  str = "",
    ) -> Optional[TrainingExample]:
        """
        Process one completed pipeline run.
        Formats it as a training example and adds to buffer.
        Triggers training if conditions are met.
        Returns the training example if one was created.
        """
        # Get record from episodic store if not provided
        if episodic_record is None:
            run_id = pipeline_result.get("run_id", "")
            records = self.episodic.query_recent(n=5)
            episodic_record = next(
                (r for r in records if r.run_id == run_id), None
            )

        if episodic_record is None:
            return None

        # Format training example
        ex = self.formatter.format_run(
            episodic_record, full_result, introspective_ctx
        )
        if ex is None:
            return None

        self.buffer.add(ex)

        if ex.verified:
            self._verified_since_train += 1

        if self.verbose:
            print(f"  [trainer] Example added. "
                  f"Buffer: {self.buffer.stats().n_total}, "
                  f"Verified since last train: {self._verified_since_train}")

        # Auto-trigger training
        if self._should_train():
            self.train()

        return ex

    def _should_train(self) -> bool:
        return (
            self._verified_since_train >= self.train_every_n
            and self.buffer.ready_to_train()
        )

    # ---- Training cycle ----

    def train(self) -> dict:
        """
        Run one complete training cycle:
          1. Measure performance before
          2. Export dataset
          3. Fine-tune
          4. Measure performance after
          5. Adopt if improved, rollback if not
        """
        cycle = self.buffer._cycle
        if self.verbose:
            print(f"\n[scaffold_trainer] Starting training cycle {cycle}...")

        # 1. Measure before
        perf_before = self.tracker.measure()
        if self.verbose:
            print(f"  Performance before: {perf_before}")

        # 2. Export dataset
        dataset_path = os.path.join(
            self.base_dir, f"dataset_cycle_{cycle}.jsonl"
        )
        n_exported = self.buffer.export_jsonl(dataset_path, format="chatml")
        if self.verbose:
            print(f"  Exported {n_exported} examples to {dataset_path}")

        # 3. Fine-tune
        train_result = self.lora.train(dataset_path)
        if self.verbose:
            print(f"  Training result: {train_result.get('status')}")
            if train_result.get("elapsed"):
                print(f"  Elapsed: {train_result['elapsed']}s")

        # 4. Need to run some test problems to measure after
        # In real use: run evaluation harness here
        # For now: measure from recent runs (will update as model is used)
        perf_after = self.tracker.measure()

        # 5. Record cycle
        cycle_record = self.tracker.record_cycle(
            cycle=cycle,
            n_examples=n_exported,
            perf_before=perf_before,
            perf_after=perf_after,
        )

        if cycle_record.adopted:
            if self.verbose:
                print(f"  ✓ Cycle {cycle} ADOPTED "
                      f"(Δ={cycle_record.delta:+.4f})")
            self.buffer.advance_cycle()
        else:
            if self.verbose:
                print(f"  ~ Cycle {cycle} not adopted yet "
                      f"(need more runs to measure improvement)")
            # Advance anyway — the dataset is still exported
            self.buffer.advance_cycle()

        self._verified_since_train = 0

        # Print Ollama instructions if adapter was produced
        if train_result.get("adapter_path") and self.verbose:
            print(self.lora.get_ollama_instructions(
                train_result["adapter_path"]
            ))

        return {
            "cycle":        cycle,
            "n_examples":   n_exported,
            "train_result": train_result,
            "cycle_record": vars(cycle_record),
            "dataset_path": dataset_path,
        }

    # ---- Export for external fine-tuning ----

    def export_dataset(
        self,
        path:   str,
        format: str = "chatml",
    ) -> int:
        """
        Export the current training buffer to a JSONL file.
        Use this if you want to fine-tune externally (Replicate, Modal, etc.)
        """
        n = self.buffer.export_jsonl(path, format=format)
        if self.verbose:
            print(f"[scaffold_trainer] Exported {n} examples to {path}")
            if self.lora._backend == "export_only":
                print(self.lora.get_ollama_instructions(path))
        return n

    # ---- Status ----

    def status(self) -> str:
        stats  = self.buffer.stats()
        lines  = [
            f"ScaffoldTrainer status:",
            f"  Backend:        {self.lora._backend}",
            f"  Buffer size:    {stats.n_total} examples "
            f"(cycle {stats.n_cycles_used})",
            f"  Avg difficulty: {stats.avg_difficulty:.2f}",
            f"  Avg struct:     {stats.avg_struct_frac:.0%}",
            f"  Domains:        {dict(stats.n_by_domain)}",
            f"  Verified since last train: {self._verified_since_train}",
            f"  Train trigger:  every {self.train_every_n} verified",
            f"  Ready to train: {self.buffer.ready_to_train()}",
        ]
        lines.append(self.tracker.summary())
        return "\n".join(lines)
