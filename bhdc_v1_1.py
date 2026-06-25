"""BHDC v1.1 - the corrected coherent core.

Faithful to the BHDC architecture working paper (Appendix A block):

    x -> norm -> encode -> {scenarios over scale x geometry on the ONE shared
    dilation spine} -> coherent superposition  Psi = sum_b c_b h_b
    -> terminal |C Psi|^2 -> GLU -> residual

Design invariants carried from the paper:
  * ONE shared dilation spine (lambda = -1/2 + i*omega); branches differ only by
    a scalar zoom s_b (= D acting) and a geometry g_b from a UNITARY bank, so the
    coherent sum is well defined.  No per-branch Hamiltonians (the v1.0 error).
  * Collapse LAST: phase is carried through the spine and geometry; the single
    Born-rule |.|^2 happens once at the end of the block.
  * Branch 0 is pinned as the identity scenario (scale = 1, geometry = I) to fix
    the phase reference, since |sum_b c_b h_b|^2 depends only on relative phases.
  * fp32 throughout (complex autograd under autocast is fragile).

Collapse arms (flags, for attribution -- see paper section 3.5):
  single / incoherent / coherent (real amp, primary) / coherent-complex;
  collapse nonlinearity |.|^2 (optical-native) vs gelu (ablation).

Self-test: ``python3 bhdc_v1_1.py``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def complex_overlap(a, b):
    return torch.sum(a.conj() * b, dim=-1)


def collision_density(psi):
    """Collision concentration of a (already unit-norm) complex state in [0, 1]."""
    p = (psi.conj() * psi).real
    d = psi.shape[-1]
    D = (p * p).sum(dim=-1)
    return (D - 1.0 / d) / (1.0 - 1.0 / d + 1e-9)


# --------------------------------------------------------------------------- #
# associative (Hillis-Steele) scan for the diagonal recurrence
#   h_t = a_t * h_{t-1} + b_t        (a_t, b_t complex, shape (..., T, d))
# Monoid combine of (older)=(a1,b1) into (newer)=(a2,b2):
#   a = a1*a2 ,  b = a2*b1 + b2
# --------------------------------------------------------------------------- #
def scan_sequential(a, b):
    """Reference O(T) scan. a, b: (M, T, d) complex. Returns h: (M, T, d)."""
    M, T, d = b.shape
    st = torch.zeros(M, d, dtype=b.dtype, device=b.device)
    outs = []
    for t in range(T):
        st = a[:, t, :] * st + b[:, t, :]
        outs.append(st)
    return torch.stack(outs, dim=1)


def scan_parallel(a, b):
    """Hillis-Steele inclusive scan, O(T log T), vectorized. Same contract as
    scan_sequential and validated against it to machine tolerance (see tests)."""
    a = a.clone()
    b = b.clone()
    T = a.shape[1]
    shift = 1
    while shift < T:
        a_sh = torch.ones_like(a)
        b_sh = torch.zeros_like(b)
        a_sh[:, shift:, :] = a[:, : T - shift, :]
        b_sh[:, shift:, :] = b[:, : T - shift, :]
        b = a * b_sh + b          # newer.b = newer.a * older.b + newer.b
        a = a_sh * a              # newer.a = older.a * newer.a
        shift *= 2
    return b


def validate_scan(M=3, T=17, d=8, tol=1e-5, seed=0):
    """Return max abs deviation between parallel and sequential scans."""
    g = torch.Generator().manual_seed(seed)
    a = torch.complex(torch.randn(M, T, d, generator=g), torch.randn(M, T, d, generator=g)) * 0.3
    b = torch.complex(torch.randn(M, T, d, generator=g), torch.randn(M, T, d, generator=g))
    err = (scan_parallel(a, b) - scan_sequential(a, b)).abs().max().item()
    return err


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
class ComplexEncoder(nn.Module):
    """x -> norm -> encode into a unit-norm complex state psi in C^d."""

    def __init__(self, in_dim, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.to_real = nn.Linear(in_dim, d_model)
        self.to_imag = nn.Linear(in_dim, d_model)

    def forward(self, x):
        x = self.norm(x)
        psi = torch.complex(self.to_real(x), self.to_imag(x))
        return psi / torch.linalg.norm(psi, dim=-1, keepdim=True).clamp_min(1e-6)


# --------------------------------------------------------------------------- #
# the one shared dilation spine
# --------------------------------------------------------------------------- #
class DilationSpine(nn.Module):
    """Diagonal scale-invariant recurrence with modes lambda = -1/2 + i*omega.
    A scalar `scale` (the zoom s_b = D acting) sets the discretization step; with
    the GR coupling on it can be a per-token tensor (density -> finer scale)."""

    def __init__(self, d_model):
        super().__init__()
        self.log_omega = nn.Parameter(torch.linspace(-2.0, 2.0, d_model))
        self.B = nn.Parameter(torch.randn(d_model, 2) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, 2) * 0.02)

    def omega(self):
        return F.softplus(self.log_omega)

    def transition(self, scale, M, T):
        """Build A = exp(lambda * scale), broadcast to (M, T, d).
        `scale` may be a python float / 0-d tensor (constant per branch) or a
        (M, T, 1) tensor (density-coupled, per token)."""
        omega = self.omega()
        lam = torch.complex(-0.5 * torch.ones_like(omega), omega)  # (d,)
        if not torch.is_tensor(scale):
            scale = torch.tensor(float(scale), device=lam.device)
        scale = scale.to(lam.real.dtype)
        if scale.dim() == 0:
            A = torch.exp(lam * scale)                       # (d,)
            return A.view(1, 1, -1).expand(M, T, -1)
        # per-token: scale is (M, T, 1)
        return torch.exp(lam.view(1, 1, -1) * scale)         # (M, T, d)

    def scan(self, psi, scale, parallel=True):
        M, T, d = psi.shape
        A = self.transition(scale, M, T)
        Bc = torch.view_as_complex(self.B.contiguous())
        b = Bc * psi
        return scan_parallel(A, b) if parallel else scan_sequential(A, b)

    def readout(self, h):
        return h * torch.view_as_complex(self.C.contiguous())

    def modular_energy(self, H):
        """<H|K|H> = sum_k omega_k |H_k|^2 -- boost generator expectation in the
        spine's own eigenbasis (used by the GR collapse-temperature coupling)."""
        w = self.omega().view(*([1] * (H.dim() - 1)), -1)
        return (w * (H.conj() * H).real).sum(dim=-1)


# --------------------------------------------------------------------------- #
# the unitary geometry bank
# --------------------------------------------------------------------------- #
class IdentityGeometry(nn.Module):
    def forward(self, psi):
        return psi


class TorusGeometry(nn.Module):
    """Maximal torus U(1)^n: per-component phase."""

    def __init__(self, d_model):
        super().__init__()
        self.phase = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, psi):
        return psi * torch.exp(1j * self.phase)


class SU2BlockGeometry(nn.Module):
    """Block-diagonal SU(2): products of Bloch-sphere rotations (unitary)."""

    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0
        self.theta = nn.Parameter(torch.randn(d_model // 2) * 0.01)
        self.phi = nn.Parameter(torch.randn(d_model // 2) * 0.01)

    def forward(self, psi):
        M, T, d = psi.shape
        x = psi.view(M, T, d // 2, 2)
        a, b = x[..., 0], x[..., 1]
        c = torch.cos(self.theta).view(1, 1, -1)
        s = torch.sin(self.theta).view(1, 1, -1)
        e = torch.exp(1j * self.phi).view(1, 1, -1)
        na = c * a - e * s * b
        nb = e.conj() * s * a + c * b
        return torch.stack([na, nb], dim=-1).reshape(M, T, d)


class MeshGeometry(nn.Module):
    """A small Reck/Clements-style mesh: a U(n) point, the optical-native primitive."""

    def __init__(self, d_model):
        super().__init__()
        self.layer1 = SU2BlockGeometry(d_model)
        self.layer2 = SU2BlockGeometry(d_model)

    def forward(self, psi):
        h = self.layer1(psi)
        h = torch.roll(h, 1, dims=-1)
        h = self.layer2(h)
        return torch.roll(h, -1, dims=-1)


def build_geometry_bank(d_model, n_branches):
    """Branch 0 is the identity reference; the rest cycle the unitary family."""
    pool = [IdentityGeometry(), TorusGeometry(d_model),
            SU2BlockGeometry(d_model), MeshGeometry(d_model)]
    geoms = [pool[0]]
    i = 1
    while len(geoms) < n_branches:
        geoms.append(pool[1 + ((i - 1) % (len(pool) - 1))])
        i += 1
    return nn.ModuleList(geoms[:n_branches])


# --------------------------------------------------------------------------- #
# GLU mixer
# --------------------------------------------------------------------------- #
class GLUMix(nn.Module):
    def __init__(self, d_model, expansion=2):
        super().__init__()
        hidden = expansion * d_model
        self.up = nn.Linear(d_model, hidden * 2)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, x):
        a, gate = self.up(x).chunk(2, dim=-1)
        return self.down(a * F.silu(gate))


# --------------------------------------------------------------------------- #
# the corrected coherent block
# --------------------------------------------------------------------------- #
COLLAPSE_MODES = ("single", "incoherent", "coherent", "coherent-complex")
COLLAPSE_NONLIN = ("abs2", "gelu")


class CoherentBlock(nn.Module):
    """encode -> scenarios over scale x geometry -> collapse -> GLU -> residual.

    All mechanisms are flags so a single combined run yields leave-one-out
    attribution (paper section 7.4 / 8).
    """

    def __init__(self, d_model, n_branches=4, collapse="coherent",
                 collapse_nonlin="abs2", expansion=2, parallel=True):
        super().__init__()
        assert collapse in COLLAPSE_MODES, collapse
        assert collapse_nonlin in COLLAPSE_NONLIN, collapse_nonlin
        assert d_model % 2 == 0
        self.d_model = d_model
        self.n_branches = n_branches
        self.collapse = collapse
        self.collapse_nonlin = collapse_nonlin
        self.parallel = parallel

        self.encoder = ComplexEncoder(d_model, d_model)
        self.spine = DilationSpine(d_model)
        self.geoms = build_geometry_bank(d_model, n_branches)

        # branch 0 zoom pinned at 1 (log_scale 0, frozen); others spread and learn.
        init = torch.linspace(-0.5, 0.5, n_branches)
        init[0] = 0.0
        self.log_scale = nn.Parameter(init)
        self._pin0 = True

        # per-token branch scores -> amplitudes c_b = sqrt(softmax(scores)).
        self.gate = nn.Linear(d_model, n_branches)
        # relative phases for the complex-amplitude arm (branch 0 pinned to 0).
        self.phase_gate = nn.Linear(d_model, n_branches)

        self.glu = GLUMix(d_model, expansion)
        self.out_norm = nn.LayerNorm(d_model)

    def scales(self):
        ls = self.log_scale
        if self._pin0:
            ls = torch.cat([ls[:1] * 0.0, ls[1:]])
        return torch.exp(ls)  # branch 0 -> 1.0

    def forward(self, x, return_telemetry=False):
        M_full = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-2], x.shape[-1]) if x.dim() > 3 else x
        B, T, d = x_flat.shape
        psi = self.encoder(x_flat)                       # (B, T, d) complex, unit norm
        scales = self.scales()

        scores = self.gate(x_flat)                       # (B, T, n_branches), real
        p = torch.softmax(scores, dim=-1)
        n = self.n_branches

        if self.collapse == "single":
            h = self.geoms[0](self.spine.scan(psi, scales[0], self.parallel))
            Cpsi = self.spine.readout(h)
            y = self._nonlin(Cpsi)
            tele = self._telemetry(p, None)
        elif self.collapse == "incoherent":
            ys, ent_acc = 0.0, []
            for b in range(n):
                h = self.geoms[b](self.spine.scan(psi, scales[b], self.parallel))
                Cpsi = self.spine.readout(h)
                ys = ys + p[..., b:b + 1] * self._nonlin(Cpsi)
                ent_acc.append(h)
            y = ys
            tele = self._telemetry(p, torch.stack(ent_acc, dim=0))
        else:  # coherent / coherent-complex
            amp = torch.sqrt(p + 1e-9)                   # (B, T, n)
            if self.collapse == "coherent-complex":
                theta = self.phase_gate(x_flat)
                theta = theta - theta[..., :1]           # branch 0 phase reference
                c = amp * torch.exp(1j * theta)          # complex amplitudes
            else:
                c = amp.to(torch.cfloat)                 # real nonnegative amplitudes
            Psi, hs = 0.0, []
            for b in range(n):
                h = self.geoms[b](self.spine.scan(psi, scales[b], self.parallel))
                Psi = Psi + c[..., b:b + 1] * h
                hs.append(h)
            Cpsi = self.spine.readout(Psi)               # |C Psi|^2: collapse once
            y = self._nonlin(Cpsi)
            tele = self._telemetry(p, torch.stack(hs, dim=0))

        out = self.out_norm(x_flat + self.glu(y))
        if x.dim() > 3:
            out = out.reshape(*M_full, T, d)
        return (out, tele) if return_telemetry else out

    def _nonlin(self, Cpsi):
        if self.collapse_nonlin == "abs2":
            return (Cpsi.conj() * Cpsi).real
        return F.gelu(Cpsi.real)

    def _telemetry(self, p, hs):
        with torch.no_grad():
            ent = -(p.clamp_min(1e-9) * p.clamp_min(1e-9).log()).sum(-1).mean()
            load = p.mean(dim=tuple(range(p.dim() - 1))).min()
            tele = {"collapse_entropy": ent.item(), "branch_load": load.item()}
            if hs is not None:
                # branch-disagreement proxy (enriched-density / coherence channel)
                tele["branch_var"] = hs.real.var(dim=0).mean().item()
        return tele


@dataclass
class BHDCConfig:
    vocab_size: int = 97
    d_model: int = 128
    n_layers: int = 2
    n_branches: int = 4
    max_len: int = 8
    collapse: str = "coherent"
    collapse_nonlin: str = "abs2"
    expansion: int = 2
    parallel: bool = True
    tie_readout: bool = False


class BHDCModel(nn.Module):
    """Token model: embed -> [CoherentBlock] x L -> head. For the grokking read
    and toy attribution runs."""

    def __init__(self, cfg: BHDCConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(cfg.max_len, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([
            CoherentBlock(cfg.d_model, cfg.n_branches, cfg.collapse,
                          cfg.collapse_nonlin, cfg.expansion, cfg.parallel)
            for _ in range(cfg.n_layers)
        ])
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, idx, return_telemetry=False):
        T = idx.shape[1]
        h = self.embed(idx) + self.pos[:T]
        teles = []
        for blk in self.blocks:
            if return_telemetry:
                h, t = blk(h, return_telemetry=True)
                teles.append(t)
            else:
                h = blk(h)
        logits = self.head(h[:, -1])                      # predict from last position
        return (logits, teles) if return_telemetry else logits


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    err = validate_scan()
    print(f"[scan] parallel vs sequential max|err| = {err:.2e}  ->  {'OK' if err < 1e-4 else 'FAIL'}")

    B, T, d = 4, 6, 16
    x = torch.randn(B, T, d)
    for mode in COLLAPSE_MODES:
        for nl in ("abs2", "gelu"):
            blk = CoherentBlock(d, n_branches=4, collapse=mode, collapse_nonlin=nl)
            out, tele = blk(x, return_telemetry=True)
            loss = out.square().mean()
            loss.backward()
            gnorm = torch.sqrt(sum(p.grad.square().sum() for p in blk.parameters() if p.grad is not None))
            ok = torch.isfinite(out).all() and torch.isfinite(gnorm)
            print(f"[block] collapse={mode:17s} nl={nl:4s} out={tuple(out.shape)} "
                  f"grad_norm={gnorm:.3f} entropy={tele['collapse_entropy']:.3f} "
                  f"load={tele['branch_load']:.3f} -> {'OK' if ok else 'FAIL'}")

    cfg = BHDCConfig(vocab_size=97, d_model=64, n_layers=2, n_branches=4, max_len=4)
    model = BHDCModel(cfg)
    idx = torch.randint(0, 97, (8, 3))
    logits = model(idx)
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,} "
          f"logits={tuple(logits.shape)} -> OK")


if __name__ == "__main__":
    _selftest()
