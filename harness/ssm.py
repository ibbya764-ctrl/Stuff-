"""Minimal pure-SSM "science arm" with spectral parametrization (v3 Part 3.2).

Design points, mapped to the v3 record:

* The dynamics operator is DIAGONAL and parametrized BY ITS SPECTRUM:
  lambda_k = -softplus(w_k) + i*nu_k.  Stability is structural (widths
  positive by construction), the SSM analogue of "resonances live in the
  lower half-plane".
* ``width_mode="critical_line"`` ties all widths to ONE trainable scalar
  per layer (the RH-shaped, maximally rigid spectrum of Part 3b.1);
  ``"free"`` is the standard control.
* ``freq_init`` chooses the frequency ladder: ``"s4"`` (S4D-Lin style),
  ``"geometric"`` (RoPE-style geometric base, the tuned control), or
  ``"golden"`` (adjacent ratios locked to phi — the Hurwitz/anti-resonance
  rider of Part 3b.2; ``"golden_lds"`` is the golden-angle low-discrepancy
  variant).
* ``gue_regularizer()`` returns a beta=2 log-repulsion penalty on the
  dynamics frequencies ONLY.  Per the two-spectral-targets doctrine it is
  never applied to trainable weight matrices.
* ``dynamics_spectrum()`` exposes the eigenvalues for spectral_telemetry.

Pure torch; the recurrence is evaluated as an FFT convolution (S4D-style
Vandermonde kernel), so it runs fine on CPU at smoke-test scale and on the
workstation GPU at experiment scale.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def make_frequencies(d_state: int, mode: str, nu_max: float = math.pi * 64) -> torch.Tensor:
    """Initial frequency ladder nu_k > 0 for one SSM layer."""
    k = torch.arange(d_state, dtype=torch.float32)
    if mode == "s4":                       # S4D-Lin: nu_k = pi * k
        return math.pi * (k + 1.0)
    if mode == "geometric":                # RoPE-style geometric ladder
        ratio = (1.0 / nu_max) ** (1.0 / max(d_state - 1, 1))
        return nu_max * ratio ** k
    if mode == "golden":                   # adjacent ratios = phi (Hurwitz rider)
        return nu_max * PHI ** (-k)
    if mode == "golden_lds":               # golden-angle low-discrepancy set
        frac = torch.remainder((k + 1.0) * (PHI - 1.0), 1.0)
        return nu_max * torch.sort(frac).values + 1e-3
    raise ValueError(f"unknown freq_init {mode!r}")


class SpectralSSMLayer(nn.Module):
    """Diagonal complex SSM, spectrum-parametrized, FFT-convolution forward."""

    def __init__(self, d_model: int, d_state: int = 64, width_mode: str = "free",
                 freq_init: str = "s4", dt: float = 1e-2, w0: float = 0.5):
        super().__init__()
        assert width_mode in ("free", "critical_line")
        self.d_model, self.d_state, self.width_mode = d_model, d_state, width_mode
        self.dt = dt
        # inverse softplus so widths start at w0
        w_raw0 = math.log(math.expm1(w0))
        if width_mode == "critical_line":
            self.w_raw = nn.Parameter(torch.tensor(w_raw0))
        else:
            self.w_raw = nn.Parameter(torch.full((d_state,), w_raw0))
        self.nu = nn.Parameter(make_frequencies(d_state, freq_init))
        self.B = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d_model))

    # -- spectrum ------------------------------------------------------
    def eigenvalues(self) -> torch.Tensor:
        """Continuous-time dynamics eigenvalues lambda = -w + i*nu."""
        w = F.softplus(self.w_raw)
        if self.width_mode == "critical_line":
            w = w.expand(self.d_state)
        return torch.complex(-w, self.nu)

    def gue_regularizer(self, eps: float = 1e-4) -> torch.Tensor:
        """beta=2 Coulomb-gas log-repulsion on the dynamics frequencies only."""
        nu = self.nu
        diff = nu[:, None] - nu[None, :]
        iu = torch.triu_indices(len(nu), len(nu), offset=1)
        pair = diff[iu[0], iu[1]].abs()
        scale = nu.detach().abs().mean() + eps
        return -torch.log(pair / scale + eps).mean()

    # -- forward -------------------------------------------------------
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (batch, L, d_model) -> (batch, L, d_model)."""
        L = u.shape[1]
        lam = self.eigenvalues()                                  # (N,)
        t = torch.arange(L, device=u.device, dtype=torch.float32)
        a_pow = torch.exp(lam.unsqueeze(1) * self.dt * t)          # (N, L)
        # kernel_d[l] = sum_n C[d,n] B[d,n] a_n^l   (real part used)
        CB = (self.C * self.B).to(a_pow.dtype)                    # (D, N)
        K = torch.einsum("dn,nl->dl", CB, a_pow).real * self.dt   # (D, L)
        # causal FFT convolution along time
        n_fft = 2 * L
        Kf = torch.fft.rfft(K, n=n_fft)                           # (D, F)
        Uf = torch.fft.rfft(u.transpose(1, 2), n=n_fft)           # (B, D, F)
        y = torch.fft.irfft(Uf * Kf, n=n_fft)[..., :L]            # (B, D, L)
        return y.transpose(1, 2) + u * self.D


class SSMBlock(nn.Module):
    def __init__(self, d_model: int, **ssm_kw):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SpectralSSMLayer(d_model, **ssm_kw)
        self.norm2 = nn.LayerNorm(d_model)
        self.glu_in = nn.Linear(d_model, 4 * d_model)
        self.glu_out = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        h = self.glu_in(self.norm2(x))
        a, b = h.chunk(2, dim=-1)
        return x + self.glu_out(a * torch.sigmoid(b))


class SpectralSSMModel(nn.Module):
    """Embedding -> n_layers x SSMBlock -> LM head.  The science arm."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4,
                 d_state: int = 64, width_mode: str = "free",
                 freq_init: str = "s4", dt: float = 1e-2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            SSMBlock(d_model, d_state=d_state, width_mode=width_mode,
                     freq_init=freq_init, dt=dt)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

    # -- character/council field bridge ---------------------------------
    def _field_token_ids(self, text: str, max_tokens: int = 128) -> torch.Tensor:
        """Deterministic word->id map into the model's own vocab.

        FNV-1a so the same text always lands on the same embedding rows across
        processes. This is the field-encoding path (hidden states are the
        object), not the generation path; a real run should graft an
        instruction-tuned generator (addendum §4.6) for judge-appraisable text.
        """
        vocab = self.embed.num_embeddings
        toks = (text.lower().replace("\n", " ").split() or ["<empty>"])[:max_tokens]
        ids = []
        for tok in toks:
            h = 2166136261
            for byte in tok.encode("utf-8", errors="ignore"):
                h ^= byte
                h = (h * 16777619) & 0xFFFFFFFF
            ids.append(h % vocab)
        dev = self.embed.weight.device
        return torch.tensor(ids, dtype=torch.long, device=dev)

    def encode_field(self, prompt: str = "", draft: str = "", dim: int | None = None) -> dict:
        """BHDC field readout for the character/council ``ExternalBHDCFieldAdapter``.

        Returns the SSM's own per-token hidden field so the conscience/council
        reads the real operator-driven state instead of the hashed demo
        embedding. ``psi`` is [tokens, d_model]; density and cognitive curvature
        follow the same conventions as the council's field adapters.

        The council must be built with ``dim == d_model`` (the adapter asserts
        it downstream); we fail loudly here rather than let a mismatch surface
        as a silent fallback.
        """
        d_model = self.embed.embedding_dim
        if dim is not None and dim != d_model:
            raise ValueError(
                f"council dim {dim} != SSM d_model {d_model}; build "
                f"BHDCGeometryCouncilModel(dim={d_model}) to match the field."
            )
        ids = self._field_token_ids(f"{prompt} {draft}")
        with torch.no_grad():
            x = self.embed(ids.unsqueeze(0))
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
        psi = x.squeeze(0)                                   # (L, d_model)
        density = psi.norm(dim=-1)
        if psi.shape[0] >= 3:
            second = psi[:-2] - 2 * psi[1:-1] + psi[2:]
            mid = second.norm(dim=-1)
            cognitive_curvature = torch.cat([mid[:1], mid, mid[-1:]], dim=0)
        else:
            cognitive_curvature = torch.ones(psi.shape[0], device=psi.device) * 0.1
        return {"psi": psi, "density": density, "cognitive_curvature": cognitive_curvature}

    # -- telemetry hooks ------------------------------------------------
    def dynamics_spectrum(self) -> np.ndarray:
        """All layers' dynamics eigenvalues, for spectral_telemetry."""
        with torch.no_grad():
            return np.concatenate(
                [b.ssm.eigenvalues().cpu().numpy() for b in self.blocks])

    def gue_regularizer(self) -> torch.Tensor:
        return torch.stack([b.ssm.gue_regularizer() for b in self.blocks]).mean()

    def weight_matrices(self) -> dict:
        """Named 2D trainable weights (for Hill-alpha reads; NEVER GUE-reg)."""
        return {n: p.detach().cpu().numpy()
                for n, p in self.named_parameters()
                if p.ndim == 2 and "embed" not in n}
