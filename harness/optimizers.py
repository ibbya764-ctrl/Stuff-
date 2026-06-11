"""Optimizers for ladder rung 1: Adam vs Muon-class vs FS/natural gradient.

The rung-1 question (v3 Part 2): when the target lives on CP^2, does
Fisher/Fubini-Study geometry buy anything beyond GENERIC spectral
conditioning?  Three arms:

  adam     -- AdamW everywhere (baseline).
  muon     -- Muon (momentum orthogonalized by Newton-Schulz = spectral-norm
              steepest descent) on hidden 2D weights, AdamW elsewhere.
              Generic spectral conditioning, no geometry.
  natgrad  -- AdamW body + exact Fubini-Study Riemannian gradient on the
              CP^2-valued output head (FS = Fisher/Bures, the program's one
              canonical geometry result).  This is the cheap exact-geometry
              instantiation; a full K-FAC/Gauss-Newton pullback is the
              heavyweight follow-up if this arm separates.

Read-out: if natgrad ~ muon >> adam, the win is generic conditioning and
FS=Fisher adds nothing in practice (kill).  If natgrad separates from muon
specifically on CP^2 targets, that is the program's first positive,
mechanism-localized geometric result.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Muon (Keller Jordan's reference algorithm, CPU/GPU float32)
# ----------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate polar factor (orthogonalization) of a 2D matrix."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon for 2D hidden weights.  Pair with AdamW for everything else."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.ndim != 2:
                    continue
                g = p.grad
                state = self.state[p]
                buf = state.setdefault("momentum_buffer", torch.zeros_like(g))
                buf.mul_(group["momentum"]).add_(g)
                d = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                O = zeropower_via_newtonschulz5(d, group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(O, alpha=-group["lr"] * scale)
        return loss


def split_param_groups(model: nn.Module):
    """(hidden 2D weights for Muon, everything else for AdamW)."""
    muon_p, other_p = [], []
    for n, p in model.named_parameters():
        (muon_p if (p.ndim == 2 and "embed" not in n and "head" not in n)
         else other_p).append(p)
    return muon_p, other_p


# ----------------------------------------------------------------------
# CP^2 target task + FS-geometry head update
# ----------------------------------------------------------------------

def normalize_state(z: torch.Tensor) -> torch.Tensor:
    """Map a complex 3-vector to a representative of its CP^2 point."""
    return z / (z.norm(dim=-1, keepdim=True) + 1e-12)


def fidelity_loss(z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """-log |<psi, phi>|^2  (FS-distance surrogate; gauge invariant)."""
    psi = normalize_state(z)
    inner = (psi.conj() * target).sum(-1)
    return -(inner.abs().pow(2) + 1e-12).log().mean()


class CP2Net(nn.Module):
    """Small MLP mapping real inputs -> a point of CP^2 (3 complex amps)."""

    def __init__(self, d_in=16, d_hidden=128, n_hidden=2):
        super().__init__()
        layers, d = [], d_in
        for _ in range(n_hidden):
            layers += [nn.Linear(d, d_hidden), nn.GELU()]
            d = d_hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d_hidden, 6)     # 3 complex amplitudes

    def states(self, x):
        h = self.head(self.body(x))
        return torch.complex(h[..., :3], h[..., 3:])


def fs_project_head_grad(net: CP2Net, x: torch.Tensor) -> None:
    """Exact FS geometry on the output: project the head gradient through
    (I - psi psi^dagger), the tangent projector of CP^2, removing the pure
    gauge/radial directions that a Euclidean step wastes effort on.

    Cheap exact-geometry version (per-batch averaged projector pullback);
    the K-FAC pullback is the follow-up if this arm separates."""
    with torch.no_grad():
        z = net.states(x)                       # (B, 3) complex
        psi = normalize_state(z)
        # average tangent projector over the batch, in the 6-real basis
        P = torch.zeros(3, 3, dtype=torch.cfloat, device=z.device)
        eye = torch.eye(3, dtype=torch.cfloat, device=z.device)
        for s in psi:
            P += eye - torch.outer(s, s.conj())
        P /= len(psi)
        # real 6x6 representation of the complex projector
        Pr = torch.zeros(6, 6, device=z.device)
        Pr[:3, :3] = P.real;  Pr[:3, 3:] = -P.imag
        Pr[3:, :3] = P.imag;  Pr[3:, 3:] = P.real
        if net.head.weight.grad is not None:
            net.head.weight.grad.copy_(Pr @ net.head.weight.grad)
        if net.head.bias.grad is not None:
            net.head.bias.grad.copy_(Pr @ net.head.bias.grad)


def make_cp2_dataset(n=4096, d_in=16, seed=0):
    """Random inputs -> deterministic random-rotation targets on CP^2."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_in, generator=g)
    w = torch.randn(d_in, 6, generator=g) / math.sqrt(d_in)
    raw = x @ w
    target = normalize_state(torch.complex(raw[:, :3], raw[:, 3:]))
    return x, target


def run_rung1(steps=2000, batch=128, seed=0, lr_adam=1e-3, lr_muon=0.02,
              log_every=200, quick=False):
    """Three-arm optimizer comparison on the CP^2 target.  Returns dict."""
    if quick:
        steps, log_every = 300, 100
    results = {}
    x_all, t_all = make_cp2_dataset(seed=seed)
    n_train = int(0.9 * len(x_all))
    for arm in ("adam", "muon", "natgrad"):
        torch.manual_seed(seed)
        net = CP2Net()
        if arm == "muon":
            muon_p, other_p = split_param_groups(net)
            opts = [Muon(muon_p, lr=lr_muon),
                    torch.optim.AdamW(other_p, lr=lr_adam)]
        else:
            opts = [torch.optim.AdamW(net.parameters(), lr=lr_adam)]
        g = torch.Generator().manual_seed(seed + 1)
        curve = []
        for step in range(steps):
            idx = torch.randint(0, n_train, (batch,), generator=g)
            loss = fidelity_loss(net.states(x_all[idx]), t_all[idx])
            for o in opts:
                o.zero_grad()
            loss.backward()
            if arm == "natgrad":
                fs_project_head_grad(net, x_all[idx])
            for o in opts:
                o.step()
            if step % log_every == 0 or step == steps - 1:
                with torch.no_grad():
                    val = fidelity_loss(net.states(x_all[n_train:]),
                                        t_all[n_train:]).item()
                curve.append((step, round(float(loss.item()), 5), round(val, 5)))
        results[arm] = {"curve": curve, "final_val": curve[-1][2],
                        "n_params": sum(p.numel() for p in net.parameters())}
    return results


if __name__ == "__main__":
    import json
    import sys
    quick = "--quick" in sys.argv
    out = run_rung1(quick=quick)
    print(json.dumps(out, indent=2))
    print("\nfinal validation loss (lower = better):")
    for arm, r in out.items():
        print(f"  {arm:8s} {r['final_val']:.5f}")
