"""Validation for the combined model and the section-8 assertion gates.

Run: ``python3 test_combined.py``  (exits non-zero on any failure).
"""
import sys
import torch
import torch.nn.functional as F
from bhdc_combined import CombinedConfig, CombinedBHDC, aggregate_telemetry, assertion_gates

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def mk(**kw):
    cfg = CombinedConfig(vocab_size=53, d_model=64, n_layers=2, n_branches=4,
                         max_len=4, entangle_k=8, **kw)
    return CombinedBHDC(cfg)


def test_all_flag_combos_train():
    idx = torch.randint(0, 53, (8, 4))
    tgt = torch.randint(0, 53, (8,))
    combos = [dict(), dict(geometry_coupling=True),
              dict(use_entanglement=True, entangle_basis="eigen"),
              dict(use_entanglement=True, entangle_basis="generic"),
              dict(geometry_coupling=True, use_entanglement=True)]
    for kw in combos:
        model = mk(**kw)
        logits, teles = model(idx, return_telemetry=True)
        loss = F.cross_entropy(logits, tgt)
        loss.backward()
        g = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))
        check(f"combined trains {kw or 'core-only'}",
              torch.isfinite(logits).all().item() and torch.isfinite(g).item(),
              f"grad={g:.2f}")


def test_flags_change_telemetry():
    idx = torch.randint(0, 53, (8, 4))
    _, t_core = mk()(idx, return_telemetry=True)
    _, t_geo = mk(geometry_coupling=True)(idx, return_telemetry=True)
    _, t_ent = mk(use_entanglement=True)(idx, return_telemetry=True)
    check("geometry flag adds density telemetry",
          "density_mean" not in t_core[0] and "density_mean" in t_geo[0])
    check("entanglement flag adds entropy telemetry",
          "entanglement_entropy" not in t_core[0] and "entanglement_entropy" in t_ent[0])


def test_entanglement_tied_to_spine():
    # eigen ladder must follow the live spine spectrum (operator in the wave function)
    model = mk(use_entanglement=True, entangle_basis="eigen")
    ent, core = model.ents[0], model.cores[0]
    lad0 = ent.ladder("cpu").clone()
    with torch.no_grad():
        core.spine.log_omega += 1.5            # perturb the spine spectrum
    lad1 = ent.ladder("cpu")
    check("eigen entangled ladder is tied to the live spine",
          (lad0 - lad1).abs().max().item() > 1e-4)


def test_gates():
    idx = torch.randint(0, 53, (8, 4))
    model = mk(use_entanglement=True)
    _, teles = model(idx, return_telemetry=True)
    # clean
    stops, warns = assertion_gates(0.5, 1.0, teles)
    check("clean step: no hard stop", len(stops) == 0)
    # non-finite loss -> hard stop
    stops, _ = assertion_gates(float("nan"), 1.0, teles)
    check("non-finite loss triggers hard stop", "non-finite loss" in stops)
    # forced product state -> warning
    teles2 = [dict(branch_load=0.2, entanglement_entropy=0.0)]
    _, warns = assertion_gates(0.5, 1.0, teles2)
    check("product-state entanglement triggers warning", any("product" in w for w in warns))
    # router collapse -> warning
    teles3 = [dict(branch_load=0.0)]
    _, warns = assertion_gates(0.5, 1.0, teles3)
    check("router collapse triggers warning", any("router" in w for w in warns))


def test_aggregate():
    idx = torch.randint(0, 53, (8, 4))
    _, teles = mk(geometry_coupling=True, use_entanglement=True)(idx, return_telemetry=True)
    agg = aggregate_telemetry(teles)
    check("aggregate has all section-8 channels",
          {"collapse_entropy", "branch_load_min", "density_mean", "entanglement_entropy"} <= set(agg))


if __name__ == "__main__":
    test_all_flag_combos_train()
    test_flags_change_telemetry()
    test_entanglement_tied_to_spine()
    test_gates()
    test_aggregate()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        sys.exit(1)
    print("ALL COMBINED CHECKS PASSED")
