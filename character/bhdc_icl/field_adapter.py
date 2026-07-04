from __future__ import annotations

"""BHDC field adapters.

The first scaffold used a deterministic hashed bag-of-words embedding in the
trainer. This module replaces that hard-coded path with an explicit field
adapter interface. A real BHDC/v17/v18 model can be attached by passing an
object with one of the supported field methods, while the demo adapter remains
small and deterministic enough for tests.

Expected real-model hooks, checked in this order:

- encode_field(prompt=..., draft=..., dim=...) -> FastFieldState | dict | tensor
- forward_field(prompt=..., draft=..., dim=...) -> FastFieldState | dict | tensor
- bdhc_field(prompt, draft) -> FastFieldState | dict | tensor
- encode_text(text) -> tensor

The adapter normalises those outputs into FastFieldState so the rest of ICL can
consume real psi/density/curvature without knowing the source model's internals.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tensor_ops import l2_normalize
from .types import FastFieldState


class FieldAdapter(Protocol):
    dim: int

    def encode(self, prompt: str, draft: str = "") -> FastFieldState:
        ...

    def candidate_vector(self, text: str, field: Optional[FastFieldState] = None) -> torch.Tensor:
        ...


@dataclass
class FieldAdapterConfig:
    dim: int = 64
    device: str | torch.device = "cpu"
    max_tokens: int = 128
    eps: float = 1e-8


def _stable_token_id(token: str, vocab_size: int) -> int:
    # Python's built-in hash is intentionally process-randomised; use a small
    # deterministic hash so traces/probes are reproducible across runs.
    h = 2166136261
    for b in token.encode("utf-8", errors="ignore"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h % vocab_size


def _tokenize(text: str, max_tokens: int) -> list[str]:
    return text.lower().replace("\n", " ").split()[:max_tokens]


class HashBHDCFieldAdapter:
    """Deterministic BHDC-shaped field adapter for tests/demos.

    It is still lightweight, but it no longer exposes only one flat hashed text
    vector. It returns a field psi[token, dim], density, and a curvature proxy,
    matching the v18 dataflow shape expected by the ICL.
    """

    def __init__(self, dim: int = 64, device: str | torch.device = "cpu", max_tokens: int = 128):
        self.config = FieldAdapterConfig(dim=dim, device=device, max_tokens=max_tokens)
        self.dim = dim
        self.device = torch.device(device)

    def _token_vec(self, token: str) -> torch.Tensor:
        v = torch.zeros(self.dim, device=self.device)
        idx = _stable_token_id(token, self.dim)
        v[idx] = 1.0
        # Add a second phase-shifted component so nearby token collisions are
        # less dominant; this mimics a tiny Fourier/scale component.
        idx2 = _stable_token_id(token[::-1] + "::phase", self.dim)
        v[idx2] += 0.5
        return l2_normalize(v)

    def encode(self, prompt: str, draft: str = "") -> FastFieldState:
        tokens = _tokenize(f"{prompt} {draft}", self.config.max_tokens)
        if not tokens:
            tokens = ["<empty>"]
        psi = torch.stack([self._token_vec(t) for t in tokens], dim=0)
        density = psi.norm(dim=-1)
        if psi.shape[0] >= 3:
            curvature_vec = psi[:-2] - 2 * psi[1:-1] + psi[2:]
            mid = curvature_vec.norm(dim=-1)
            cognitive_curvature = torch.cat([mid[:1], mid, mid[-1:]], dim=0)
        else:
            cognitive_curvature = torch.ones_like(density) * 0.1
        return FastFieldState(
            psi=psi,
            density=density,
            cognitive_curvature=cognitive_curvature,
            draft_text=draft,
        )

    def candidate_vector(self, text: str, field: Optional[FastFieldState] = None) -> torch.Tensor:
        if field is None:
            field = self.encode("", text)
        weights = (field.density * (1.0 + field.cognitive_curvature)).detach()
        pooled = (field.psi * weights[:, None]).sum(dim=0) / (weights.sum() + self.config.eps)
        return l2_normalize(pooled)


class TorchBHDCFieldAdapter(nn.Module):
    """Small trainable BHDC-style field encoder.

    This is a practical bridge between the pure scaffold and a real BHDC model:
    it produces a continuous field, density and curvature from token ids, while
    remaining independent from any specific external LM. It can later be swapped
    for the actual CP²/BHDC field stack without changing the ICL controller.
    """

    def __init__(
        self,
        dim: int = 128,
        vocab_size: int = 8192,
        device: str | torch.device = "cpu",
        max_tokens: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.device = torch.device(device)
        self.embed = nn.Embedding(vocab_size, dim)
        self.scale_gate = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.curv_proj = nn.Linear(dim, 1)
        self.to(self.device)

    def token_ids(self, text: str) -> torch.Tensor:
        toks = _tokenize(text, self.max_tokens) or ["<empty>"]
        ids = [_stable_token_id(t, self.vocab_size) for t in toks]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def encode(self, prompt: str, draft: str = "") -> FastFieldState:
        ids = self.token_ids(f"{prompt} {draft}")
        raw = self.embed(ids)
        psi = l2_normalize(raw + 0.1 * self.scale_gate(raw))
        density = torch.sigmoid(psi.norm(dim=-1))
        if psi.shape[0] >= 3:
            second = psi[:-2] - 2 * psi[1:-1] + psi[2:]
            curv_mid = F.softplus(self.curv_proj(second).squeeze(-1))
            cognitive_curvature = torch.cat([curv_mid[:1], curv_mid, curv_mid[-1:]], dim=0)
        else:
            cognitive_curvature = F.softplus(self.curv_proj(psi).squeeze(-1))
        return FastFieldState(
            psi=psi,
            density=density,
            cognitive_curvature=cognitive_curvature,
            draft_text=draft,
        )

    def candidate_vector(self, text: str, field: Optional[FastFieldState] = None) -> torch.Tensor:
        if field is None:
            field = self.encode("", text)
        weights = field.density * (1.0 + field.cognitive_curvature)
        pooled = (field.psi * weights[:, None]).sum(dim=0) / (weights.sum() + 1e-8)
        return l2_normalize(pooled)


class ExternalBHDCFieldAdapter:
    """Adapter for an existing BHDC/CP² field model.

    The external model is deliberately duck-typed to avoid forcing your current
    codebase into this package's class hierarchy.
    """

    def __init__(
        self,
        model: Any,
        dim: int,
        device: str | torch.device = "cpu",
        fallback: Optional[FieldAdapter] = None,
    ):
        self.model = model
        self.dim = dim
        self.device = torch.device(device)
        self.fallback = fallback or HashBHDCFieldAdapter(dim=dim, device=device)

    def _call_model(self, prompt: str, draft: str) -> Any:
        for name in ("encode_field", "forward_field"):
            fn = getattr(self.model, name, None)
            if callable(fn):
                try:
                    return fn(prompt=prompt, draft=draft, dim=self.dim)
                except TypeError:
                    return fn(prompt, draft)
        fn = getattr(self.model, "bdhc_field", None)
        if callable(fn):
            return fn(prompt, draft)
        fn = getattr(self.model, "encode_text", None)
        if callable(fn):
            return fn(f"{prompt}\n{draft}")
        raise AttributeError(
            "External BHDC model needs encode_field, forward_field, bdhc_field, or encode_text."
        )

    def _coerce(self, out: Any, draft: str) -> FastFieldState:
        if isinstance(out, FastFieldState):
            return out
        if isinstance(out, dict):
            psi = None
            for key in ("psi", "field", "hidden"):
                if key in out and out[key] is not None:
                    psi = out[key]
                    break
            if psi is None:
                raise ValueError("field dict must contain psi, field, or hidden")
            psi = torch.as_tensor(psi, device=self.device, dtype=torch.float32)
            if psi.ndim == 1:
                psi = psi[None, :]
            density = out.get("density")
            density = torch.as_tensor(density, device=self.device, dtype=torch.float32) if density is not None else psi.norm(dim=-1)
            curv = out.get("cognitive_curvature", None)
            if curv is None:
                curv = out.get("curvature", None)
            curv = torch.as_tensor(curv, device=self.device, dtype=torch.float32) if curv is not None else self._curvature_from_psi(psi)
            return FastFieldState(psi=psi, density=density, cognitive_curvature=curv, draft_text=draft)
        tensor = torch.as_tensor(out, device=self.device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor[None, :]
        return FastFieldState(
            psi=tensor,
            density=tensor.norm(dim=-1),
            cognitive_curvature=self._curvature_from_psi(tensor),
            draft_text=draft,
        )

    def _curvature_from_psi(self, psi: torch.Tensor) -> torch.Tensor:
        if psi.shape[0] >= 3:
            second = psi[:-2] - 2 * psi[1:-1] + psi[2:]
            mid = second.norm(dim=-1)
            return torch.cat([mid[:1], mid, mid[-1:]], dim=0)
        return torch.ones(psi.shape[0], device=psi.device) * 0.1

    def encode(self, prompt: str, draft: str = "") -> FastFieldState:
        try:
            return self._coerce(self._call_model(prompt, draft), draft=draft)
        except Exception:
            # Keep the wrapper usable while the external model is being wired.
            return self.fallback.encode(prompt, draft)

    def candidate_vector(self, text: str, field: Optional[FastFieldState] = None) -> torch.Tensor:
        if field is None:
            field = self.encode("", text)
        if field.psi.shape[-1] != self.dim:
            raise ValueError(f"field dim {field.psi.shape[-1]} != adapter dim {self.dim}")
        weights = field.density * (1.0 + field.cognitive_curvature)
        pooled = (field.psi * weights[:, None]).sum(dim=0) / (weights.sum() + 1e-8)
        return l2_normalize(pooled.detach())
