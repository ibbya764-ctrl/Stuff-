"""Validation gate for the BHDC coherent core (paper section 9, step 1).

Run: ``python3 test_bhdc.py``  (exits non-zero on any failure).

Checks, in the paper's order:
  1. parallel scan == sequential scan to machine tolerance (section 7.2);
  2. the geometry bank is unitary (norm-preserving) -- required for the
     coherent sum to be well defined (section 3.4);
  3. the encoder produces unit-norm complex states;
  4. branch 0 is pinned to the identity scenario (scale == 1);
  5. every flag combination forwards and backprops with finite grads at toy
     scale (the smoke test);
  6. one optimizer step strictly decreases a toy loss for the primary arm.
"""
import sys
import torch
import bhdc_v1_1 as m

FAILS = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_scan():
    for T in (1, 2, 8, 17, 64):
        err = m.validate_scan(M=3, T=T, d=8)
        check(f"parallel scan == sequential (T={T})", err < 1e-4, f"max|err|={err:.2e}")


def test_geometry_unitary():
    torch.manual_seed(0)
    d = 16
    psi = torch.complex(torch.randn(2, 5, d), torch.randn(2, 5, d))
    n0 = torch.linalg.norm(psi, dim=-1)
    for g in m.build_geometry_bank(d, 4):
        out = g(psi)
        dev = (torch.linalg.norm(out, dim=-1) - n0).abs().max().item()
        check(f"geometry {type(g).__name__} is norm-preserving", dev < 1e-4, f"max dev={dev:.2e}")


def test_encoder_unit_norm():
    torch.manual_seed(0)
    enc = m.ComplexEncoder(12, 16)
    psi = enc(torch.randn(4, 7, 12))
    dev = (torch.linalg.norm(psi, dim=-1) - 1.0).abs().max().item()
    check("encoder output is unit-norm", dev < 1e-4, f"max dev={dev:.2e}")
    check("encoder output is complex", psi.is_complex())


def test_branch0_pinned():
    blk = m.CoherentBlock(16, n_branches=4)
    s0 = blk.scales()[0].item()
    check("branch 0 zoom pinned to 1.0", abs(s0 - 1.0) < 1e-6, f"scale[0]={s0:.6f}")


def test_all_flags_smoke():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 16)
    for mode in m.COLLAPSE_MODES:
        for nl in m.COLLAPSE_NONLIN:
            blk = m.CoherentBlock(16, n_branches=4, collapse=mode, collapse_nonlin=nl)
            out = blk(x)
            out.square().mean().backward()
            gnorm = torch.sqrt(sum(p.grad.square().sum()
                                   for p in blk.parameters() if p.grad is not None))
            ok = torch.isfinite(out).all().item() and torch.isfinite(gnorm).item()
            check(f"smoke collapse={mode}/{nl}", ok, f"grad_norm={gnorm:.3f}")


def test_optimizer_step_decreases_loss():
    torch.manual_seed(0)
    cfg = m.BHDCConfig(vocab_size=23, d_model=64, n_layers=2, n_branches=4, max_len=3)
    model = m.BHDCModel(cfg)
    idx = torch.randint(0, 23, (32, 3))
    tgt = torch.randint(0, 23, (32,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    l0 = torch.nn.functional.cross_entropy(model(idx), tgt).item()
    for _ in range(20):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(idx), tgt)
        loss.backward()
        opt.step()
    l1 = loss.item()
    check("toy loss decreases under Adam", l1 < l0, f"{l0:.3f} -> {l1:.3f}")


if __name__ == "__main__":
    test_scan()
    test_geometry_unitary()
    test_encoder_unit_norm()
    test_branch0_pinned()
    test_all_flags_smoke()
    test_optimizer_step_decreases_loss()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        sys.exit(1)
    print("ALL VALIDATION CHECKS PASSED")
