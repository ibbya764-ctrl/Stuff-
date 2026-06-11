"""
local_model.py
==============

Local model backends that drop into the pipeline's `llm_chat_fn` slot.

Provides a uniform interface — `chat(system, user) -> str` — backed by
any of three options that can be swapped without touching the rest of
the system:

    - LocalLlamaCpp     : llama.cpp via the Python bindings (Q4/Q5 GGUF)
    - LocalVLLM         : vLLM server (NVFP4, faster, server-mode)
    - HostedAPI         : OpenAI/Anthropic-compatible HTTP (fallback)
    - HybridRouter      : route by call type — local for cheap calls,
                          hosted for hard reasoning

When the RTX 5070 arrives, the recommended starting setup is:

    LocalLlamaCpp(
        model_path="qwen2.5-7b-instruct-q4_k_m.gguf",
        n_gpu_layers=-1,       # offload everything to GPU
        n_ctx=8192,
        seed=0,
    )

For Qwen3 hybrid (thinking + non-thinking modes), see the model card
for the right prompt format and pass `mode="thinking"` per call.

DEPENDENCIES (lazy-imported so this module loads on any system):
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    pip install vllm                # optional
    pip install openai anthropic    # for HostedAPI / HybridRouter
"""

from __future__ import annotations

import os
import json
import time
import threading
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


# ============================================================
# Protocol: anything with this shape can plug into pipeline.py
# ============================================================

class ChatBackend(Protocol):
    """The minimal interface the pipeline expects."""

    def chat(self, system: str, user: str, **kwargs) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def is_local(self) -> bool: ...


# ============================================================
# Telemetry — every call is timed and logged
# ============================================================

@dataclass
class CallRecord:
    backend: str
    system_chars: int
    user_chars: int
    response_chars: int
    elapsed_seconds: float
    success: bool
    error: str = ""
    timestamp: float = field(default_factory=time.time)
    tag: str = ""           # caller can label calls (e.g. "branch_gen")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class TelemetryLog:
    """Threadsafe append-only call log, persists to JSON."""

    def __init__(self, path: str = "./local_model_telemetry.json"):
        self.path = path
        self._lock = threading.Lock()
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                self._records = json.load(f).get("calls", [])
        except (json.JSONDecodeError, OSError):
            self._records = []

    def record(self, rec: CallRecord) -> None:
        with self._lock:
            self._records.append(rec.to_dict())
            try:
                with open(self.path, "w") as f:
                    json.dump({"calls": self._records}, f, indent=2)
            except OSError:
                pass

    def summary(self) -> dict:
        if not self._records:
            return {"n_calls": 0}
        n = len(self._records)
        n_success = sum(1 for r in self._records if r.get("success"))
        total_time = sum(r.get("elapsed_seconds", 0) for r in self._records)
        by_backend: dict[str, int] = {}
        for r in self._records:
            by_backend[r["backend"]] = by_backend.get(r["backend"], 0) + 1
        return {
            "n_calls":          n,
            "n_success":        n_success,
            "success_rate":     round(n_success / n, 3),
            "total_time_s":     round(total_time, 1),
            "mean_time_s":      round(total_time / n, 3),
            "calls_by_backend": by_backend,
        }


# ============================================================
# Backend 1: llama.cpp (the recommended starting point for 5070)
# ============================================================

class LocalLlamaCpp:
    """
    Backend using llama-cpp-python. Designed for Q4_K_M GGUF models
    on the RTX 5070 (12 GB). Defaults are tuned for Qwen2.5-7B-Instruct
    but work for any compatible GGUF.

    Set `model_path` to a local file OR leave it as the sentinel
    "STUB" to use a deterministic stub that returns a canned response —
    useful for testing the rest of the pipeline pre-GPU.
    """

    def __init__(
        self,
        model_path: str = "STUB",
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        n_threads: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        seed: int = 0,
        telemetry: Optional[TelemetryLog] = None,
        chat_format: str = "qwen",
        verbose: bool = False,
    ):
        self.model_path   = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx        = n_ctx
        self.n_threads    = n_threads
        self.temperature  = temperature
        self.top_p        = top_p
        self.max_tokens   = max_tokens
        self.seed         = seed
        self.chat_format  = chat_format
        self.verbose      = verbose
        self.telemetry    = telemetry
        self._llm         = None
        self._is_stub     = (model_path == "STUB")

    @property
    def name(self) -> str:
        return f"llama_cpp({os.path.basename(self.model_path)})"

    @property
    def is_local(self) -> bool:
        return True

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        if self._is_stub:
            return
        try:
            from llama_cpp import Llama   # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install with:\n"
                "  pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cu124"
            ) from e
        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            seed=self.seed,
            chat_format=self.chat_format,
            verbose=self.verbose,
        )

    def chat(
        self,
        system: str,
        user: str,
        *,
        tag: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        t0 = time.time()
        response_text = ""
        success = False
        error = ""
        try:
            if self._is_stub:
                response_text = _stub_response(system, user)
                success = True
            else:
                self._ensure_loaded()
                completion = self._llm.create_chat_completion(   # type: ignore
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=(temperature if temperature is not None
                                 else self.temperature),
                    top_p=self.top_p,
                    max_tokens=(max_tokens if max_tokens is not None
                                else self.max_tokens),
                    stop=stop,
                )
                response_text = completion["choices"][0]["message"]["content"]
                success = True
        except Exception as e:
            error = str(e)
            response_text = ""
        finally:
            elapsed = time.time() - t0
            if self.telemetry is not None:
                self.telemetry.record(CallRecord(
                    backend=self.name,
                    system_chars=len(system),
                    user_chars=len(user),
                    response_chars=len(response_text),
                    elapsed_seconds=elapsed,
                    success=success,
                    error=error,
                    tag=tag,
                ))
        if not success and error:
            raise RuntimeError(f"LocalLlamaCpp call failed: {error}")
        return response_text


# ============================================================
# Backend 2: vLLM (faster, NVFP4 on Blackwell, server-mode)
# ============================================================

class LocalVLLM:
    """
    Backend talking to a vLLM HTTP server. Higher throughput than
    llama.cpp for batch operations and NVFP4-quantized models.

    Recommended startup for the 5070:
        vllm serve Qwen/Qwen2.5-7B-Instruct \\
            --dtype bfloat16 --max-model-len 8192 \\
            --gpu-memory-utilization 0.85 --port 8000

    For NVFP4 (Blackwell-native):
        vllm serve nvidia/Qwen2.5-7B-Instruct-NVFP4 \\
            --quantization modelopt --port 8000
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        telemetry: Optional[TelemetryLog] = None,
        timeout: float = 120.0,
    ):
        self.endpoint    = endpoint
        self.model       = model
        self.api_key     = api_key
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.telemetry   = telemetry
        self.timeout     = timeout

    @property
    def name(self) -> str:
        return f"vllm({self.model})"

    @property
    def is_local(self) -> bool:
        return True

    def chat(
        self,
        system: str,
        user: str,
        *,
        tag: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        import urllib.request
        import urllib.error
        t0 = time.time()
        response_text = ""
        success = False
        error = ""
        try:
            payload = json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "temperature": (temperature if temperature is not None
                                else self.temperature),
                "max_tokens": (max_tokens if max_tokens is not None
                               else self.max_tokens),
                "stop": stop or [],
            }).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint.rstrip("/") + "/chat/completions",
                data=payload,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
            response_text = body["choices"][0]["message"]["content"]
            success = True
        except Exception as e:
            error = str(e)
        finally:
            elapsed = time.time() - t0
            if self.telemetry is not None:
                self.telemetry.record(CallRecord(
                    backend=self.name,
                    system_chars=len(system),
                    user_chars=len(user),
                    response_chars=len(response_text),
                    elapsed_seconds=elapsed,
                    success=success,
                    error=error,
                    tag=tag,
                ))
        if not success:
            raise RuntimeError(f"vLLM call failed: {error}")
        return response_text


# ============================================================
# Backend 3: hosted API (fallback / hard problems)
# ============================================================

class HostedAPI:
    """
    Hosted-API backend, OpenAI-compatible. Used for:
      - hard reasoning calls when the local model is not enough
      - testing the pipeline before the GPU arrives
      - calls that need very long context

    Reads ANTHROPIC_API_KEY or OPENAI_API_KEY from the environment.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        telemetry: Optional[TelemetryLog] = None,
        timeout: float = 120.0,
    ):
        self.provider    = provider.lower()
        self.model       = model
        self.api_key     = api_key or os.environ.get(
            "ANTHROPIC_API_KEY" if self.provider == "anthropic"
            else "OPENAI_API_KEY",
            "",
        )
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.telemetry   = telemetry
        self.timeout     = timeout

    @property
    def name(self) -> str:
        return f"hosted_{self.provider}({self.model})"

    @property
    def is_local(self) -> bool:
        return False

    def chat(
        self,
        system: str,
        user: str,
        *,
        tag: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        t0 = time.time()
        response_text = ""
        success = False
        error = ""
        try:
            if self.provider == "anthropic":
                response_text = self._chat_anthropic(
                    system, user,
                    temperature or self.temperature,
                    max_tokens or self.max_tokens,
                )
            elif self.provider == "openai":
                response_text = self._chat_openai(
                    system, user,
                    temperature or self.temperature,
                    max_tokens or self.max_tokens,
                )
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            success = True
        except Exception as e:
            error = str(e)
        finally:
            elapsed = time.time() - t0
            if self.telemetry is not None:
                self.telemetry.record(CallRecord(
                    backend=self.name,
                    system_chars=len(system),
                    user_chars=len(user),
                    response_chars=len(response_text),
                    elapsed_seconds=elapsed,
                    success=success,
                    error=error,
                    tag=tag,
                ))
        if not success:
            raise RuntimeError(f"HostedAPI call failed: {error}")
        return response_text

    def _chat_anthropic(self, system, user, temp, max_tokens):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install anthropic") from e
        client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temp,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    def _chat_openai(self, system, user, temp, max_tokens):
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("pip install openai") from e
        client = openai.OpenAI(api_key=self.api_key, timeout=self.timeout)
        comp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temp,
            max_tokens=max_tokens,
        )
        return comp.choices[0].message.content or ""


# ============================================================
# Backend 4: HybridRouter — local for cheap, hosted for hard
# ============================================================

# Call tags that should be routed to the hosted (expensive) backend.
# Everything else stays local.
DEFAULT_HOSTED_TAGS: set[str] = {
    "branch_gen_hard",      # hard problem branches
    "structure_extract",    # SME extraction needs high reliability
    "metacog_audit_hard",   # adversarial audit on hard problems
}


class HybridRouter:
    """
    Routes each call to local or hosted based on the `tag` argument.

    Default routing:
        - Tag in DEFAULT_HOSTED_TAGS → hosted
        - Anything else              → local

    Override per-call with `force_backend="local"` or `"hosted"`.
    """

    def __init__(
        self,
        local: ChatBackend,
        hosted: ChatBackend,
        hosted_tags: Optional[set[str]] = None,
        telemetry: Optional[TelemetryLog] = None,
    ):
        self.local       = local
        self.hosted      = hosted
        self.hosted_tags = hosted_tags or DEFAULT_HOSTED_TAGS
        self.telemetry   = telemetry

    @property
    def name(self) -> str:
        return f"hybrid(local={self.local.name}, hosted={self.hosted.name})"

    @property
    def is_local(self) -> bool:
        return False   # hybrid by definition

    def chat(
        self,
        system: str,
        user: str,
        *,
        tag: str = "",
        force_backend: Optional[str] = None,
        **kwargs,
    ) -> str:
        use_hosted = (
            force_backend == "hosted"
            or (force_backend is None and tag in self.hosted_tags)
        )
        backend = self.hosted if use_hosted else self.local
        return backend.chat(system, user, tag=tag, **kwargs)


# ============================================================
# Pipeline adapter — convert a backend into a `(system, user) -> str`
# ============================================================

def make_llm_chat_fn(
    backend: ChatBackend,
    default_tag: str = "",
) -> Callable[[str, str], str]:
    """
    Wraps a backend as the simple `llm_chat_fn(system, user) -> str`
    callable that pipeline.py expects.

    Telemetry/tagging is preserved internally but invisible to the
    pipeline.
    """
    def _chat(system: str, user: str) -> str:
        return backend.chat(system, user, tag=default_tag)
    _chat.backend = backend       # type: ignore
    _chat.name    = backend.name  # type: ignore
    return _chat


# ============================================================
# Stub for pre-GPU testing
# ============================================================

def _stub_response(system: str, user: str) -> str:
    """
    Deterministic canned response. Used when model_path="STUB" so the
    rest of the pipeline can be exercised without a real model.

    Returns plausible-looking JSON for the three commonest pipeline
    call types, otherwise a generic acknowledgement.
    """
    s = (system + " " + user).lower()
    if "json" in s and "branches" in s:
        return '''```json
{
  "branches": [
    {
      "name": "branch_alpha",
      "method": "[stub] symbolic derivation from first principles",
      "steps": [
        "[stub] step 1: set up the governing equation",
        "[stub] step 2: apply boundary conditions",
        "[stub] step 3: solve symbolically"
      ],
      "assumptions": ["[stub] linearity holds", "[stub] domain is bounded"],
      "candidate_result": "[stub] candidate_result_alpha"
    },
    {
      "name": "branch_beta",
      "method": "[stub] perturbative expansion in small parameter",
      "steps": [
        "[stub] step 1: identify small parameter",
        "[stub] step 2: expand to leading order"
      ],
      "assumptions": ["[stub] small parameter regime"],
      "candidate_result": "[stub] candidate_result_beta"
    }
  ]
}
```'''
    if "questions" in s and ("audit" in s or "critique" in s):
        return '''{
  "questions": [
    {"question": "[stub] How is the closure relation justified?",
     "targets": "assumption_1"}
  ]
}'''
    if "entities" in s and "relations" in s:
        return '''{
  "entities": [
    {"id": "x", "type": "VARIABLE", "label": "[stub] primary variable"},
    {"id": "y", "type": "VARIABLE", "label": "[stub] secondary variable"}
  ],
  "relations": [
    {"type": "DEPENDS_ON", "args": ["x", "y"], "label": ""}
  ]
}'''
    return ("[stub local model response. Configure a real model path "
            "to get real output.]")


# ============================================================
# Convenience: one-line setup for common configurations
# ============================================================

def setup_stub(verbose: bool = False) -> tuple[Callable, TelemetryLog]:
    """Pre-GPU setup: stub backend with telemetry. Use this to exercise
    the pipeline before the RTX 5070 arrives."""
    tel = TelemetryLog("./local_model_telemetry.json")
    backend = LocalLlamaCpp(model_path="STUB", telemetry=tel, verbose=verbose)
    return make_llm_chat_fn(backend), tel


def setup_local_qwen(
    model_path: str,
    telemetry_path: str = "./local_model_telemetry.json",
    n_ctx: int = 8192,
) -> tuple[Callable, TelemetryLog]:
    """Local-only setup, Qwen2.5-7B-Instruct GGUF. Recommended starting
    point for the RTX 5070."""
    tel = TelemetryLog(telemetry_path)
    backend = LocalLlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        telemetry=tel,
        chat_format="qwen",
    )
    return make_llm_chat_fn(backend), tel


def setup_hybrid(
    local_model_path: str,
    hosted_provider: str = "anthropic",
    hosted_model: str = "claude-sonnet-4-6",
    telemetry_path: str = "./local_model_telemetry.json",
) -> tuple[Callable, TelemetryLog]:
    """Hybrid setup: local for cheap calls, hosted for hard ones."""
    tel = TelemetryLog(telemetry_path)
    local = LocalLlamaCpp(
        model_path=local_model_path,
        n_gpu_layers=-1,
        telemetry=tel,
    )
    hosted = HostedAPI(
        provider=hosted_provider,
        model=hosted_model,
        telemetry=tel,
    )
    router = HybridRouter(local=local, hosted=hosted, telemetry=tel)
    return make_llm_chat_fn(router), tel
