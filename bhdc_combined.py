"""BHDC combined model (paper section 7.1, third file) -- the canonical training
artifact: coherent core + GR geometry coupling + entangled sector behind
per-component flags, with telemetry and assertion gates (section 8).

Per layer:  h <- CoherentBlock(h)               (collapse arms, optional section-4 coupling)
            h <- h + EntangledBlock(h)           (optional section-5 pathway, residual)
The entangled block's operator-eigenbasis ladder is tied LIVE to that layer's
spine spectrum (the 'operator in the wave function' -- section 5.4).

Attribution is cheap because every mechanism is a flag (section 7.4): a few
combined-minus-one runs give leave-one-out attribution out of one job.

Self-test: ``python3 bhdc_combined.py``.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch, torch.nn as nn

from bhdc_v1_1 import CoherentBlock, COLLAPSE_MODES
from bhdc_entangled import EntangledBlock


@dataclass
class CombinedConfig:
    vocab_size: int = 97
    d_model: int = 128
    n_layers: int = 2
    n_branches: int = 4
    max_len: int = 8
    collapse: str = "coherent"
    collapse_nonlin: str = "abs2"
    expansion: int = 2
    parallel: bool = True
    # per-component flags (section 7.4)
    geometry_coupling: bool = False
    use_entanglement: bool = False
    entangle_k: int = 8
    entangle_basis: str = "eigen"          # 'eigen' (operator) | 'generic' (ablation)


class CombinedBHDC(nn.Module):
    def __init__(self, cfg: CombinedConfig):
        super().__init__()
        assert cfg.collapse in COLLAPSE_MODES, cfg.collapse
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(cfg.max_len, cfg.d_model) * 0.02)

        self.cores = nn.ModuleList()
        self.ents = nn.ModuleList()
        for _ in range(cfg.n_layers):
            core = CoherentBlock(cfg.d_model, cfg.n_branches, cfg.collapse,
                                 cfg.collapse_nonlin, cfg.expansion, cfg.parallel,
                                 cfg.geometry_coupling)
            self.cores.append(core)
            if cfg.use_entanglement:
                ent = EntangledBlock(cfg.d_model, cfg.entangle_k, cfg.entangle_basis)
                if cfg.entangle_basis == "eigen":
                    ent.set_omega_source(core.spine.omega)   # operator in the wave function
                self.ents.append(ent)
            else:
                self.ents.append(None)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, idx, return_telemetry=False):
        T = idx.shape[1]
        h = self.embed(idx) + self.pos[:T]
        teles = []
        for core, ent in zip(self.cores, self.ents):
            if return_telemetry:
                h, t = core(h, return_telemetry=True)
            else:
                h = core(h)
                t = {}
            if ent is not None:
                if return_telemetry:
                    e_out, e_t = ent(h, return_telemetry=True)
                    t["entanglement_entropy"] = e_t["entanglement_entropy"]
                    t["entanglement_entropy_max"] = e_t["entanglement_entropy_max"]
                else:
                    e_out = ent(h)
                h = h + e_out
            teles.append(t)
        logits = self.head(h[:, -1])
        return (logits, teles) if return_telemetry else logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def aggregate_telemetry(teles):
    """Collapse per-layer telemetry into the section-8 channels."""
    agg = {}
    keys = set().union(*[t.keys() for t in teles]) if teles else set()
    if "collapse_entropy" in keys:
        agg["collapse_entropy"] = sum(t["collapse_entropy"] for t in teles) / len(teles)
    if "branch_load" in keys:
        agg["branch_load_min"] = min(t["branch_load"] for t in teles if "branch_load" in t)
    if "density_mean" in keys:
        dm = [t["density_mean"] for t in teles if "density_mean" in t]
        agg["density_mean"] = sum(dm) / len(dm)
    if "entanglement_entropy" in keys:
        es = [t["entanglement_entropy"] for t in teles if "entanglement_entropy" in t]
        agg["entanglement_entropy"] = sum(es) / len(es)
    return agg


def assertion_gates(loss, grad_norm, teles, ent_floor=0.05, load_floor=1e-3):
    """Section 8 gates. Returns (stops, warnings).
    Hard stop: non-finite loss / grad. Warnings: entanglement -> product state,
    router (branch-load) collapse."""
    stops, warns = [], []
    if not torch.isfinite(torch.as_tensor(loss)):
        stops.append("non-finite loss")
    if grad_norm is not None and not torch.isfinite(torch.as_tensor(grad_norm)):
        stops.append("non-finite grad norm")
    for i, t in enumerate(teles):
        if "entanglement_entropy" in t and t["entanglement_entropy"] < ent_floor:
            warns.append(f"layer{i}: entanglement decayed to product state "
                         f"(S={t['entanglement_entropy']:.3f} < {ent_floor})")
        if "branch_load" in t and t["branch_load"] < load_floor:
            warns.append(f"layer{i}: router collapse (min branch load "
                         f"{t['branch_load']:.1e} < {load_floor})")
    return stops, warns


def _selftest():
    torch.manual_seed(0)
    idx = torch.randint(0, 97, (8, 4))
    tgt = torch.randint(0, 97, (8,))
    configs = {
        "core-only":        dict(),
        "+geometry":        dict(geometry_coupling=True),
        "+entangle(eigen)": dict(use_entanglement=True, entangle_basis="eigen"),
        "+entangle(gen)":   dict(use_entanglement=True, entangle_basis="generic"),
        "all-on":           dict(geometry_coupling=True, use_entanglement=True),
    }
    for name, kw in configs.items():
        cfg = CombinedConfig(vocab_size=97, d_model=64, n_layers=2, n_branches=4,
                             max_len=4, entangle_k=8, **kw)
        model = CombinedBHDC(cfg)
        logits, teles = model(idx, return_telemetry=True)
        loss = nn.functional.cross_entropy(logits, tgt)
        loss.backward()
        gnorm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))
        stops, warns = assertion_gates(loss.item(), gnorm.item(), teles)
        agg = aggregate_telemetry(teles)
        ok = torch.isfinite(logits).all() and torch.isfinite(gnorm) and not stops
        agg_str = " ".join(f"{k}={v:.3f}" for k, v in agg.items())
        print(f"[{name:17s}] params={model.n_params():>7,} grad={gnorm:5.2f} "
              f"| {agg_str} -> {'OK' if ok else 'FAIL'}")
        if warns:
            print(f"   warnings: {warns}")


if __name__ == "__main__":
    _selftest()
