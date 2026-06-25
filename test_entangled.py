"""Validation for the entangled two-sector block (paper section 5).

Run: ``python3 test_entangled.py``  (exits non-zero on any failure).

Checks the numerics the paper flags as the fiddliest / least-validated code:
  1. ||M||_F == 1 -- the Schmidt construction is bounded (no SVD, no norm blow-up);
  2. r = 0 -> product state (S ~ 0); r large -> entangled (S -> log K);
  3. entanglement entropy is monotone in r;
  4. U_A, U_B are unitary (matrix-exp of an anti-Hermitian generator);
  5. both bases (operator-eigen / generic) forward and backprop with finite grads;
  6. the reduced state eigenvalues equal the Schmidt occupations p_n.
"""
import sys
import math
import torch
import bhdc_entangled as e

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_bounded_norm():
    torch.manual_seed(0)
    blk = e.EntangledBlock(16, k=8, basis="eigen")
    for r in (0.0, 0.5, 1.0, 3.0):
        M, p, _, _ = blk.joint_state(torch.tensor([r]))
        fro = torch.linalg.norm(M.reshape(-1)).item()
        check(f"||M||_F == 1 at r={r}", abs(fro - 1) < 1e-4, f"{fro:.6f}")
        check(f"sum p == 1 at r={r}", abs(p.sum().item() - 1) < 1e-4)


def test_product_to_entangled():
    for basis in ("eigen", "generic"):
        blk = e.EntangledBlock(16, k=8, basis=basis)
        lad = blk.ladder("cpu")
        Smax = math.log(8)
        S0 = e.entanglement_entropy(e.schmidt_occupation(torch.tensor(0.0), lad)).item()
        Sbig = e.entanglement_entropy(e.schmidt_occupation(torch.tensor(4.0), lad)).item()
        check(f"{basis}: r=0 is product (S<2% max)", S0 < 0.02 * Smax, f"S0={S0:.2e}")
        check(f"{basis}: r large is entangled (S>0.5 max)", Sbig > 0.5 * Smax, f"S={Sbig:.3f}")


def test_monotone_entropy():
    blk = e.EntangledBlock(16, k=8, basis="generic")
    lad = blk.ladder("cpu")
    rs = torch.linspace(0.0, 4.0, 9)
    S = torch.stack([e.entanglement_entropy(e.schmidt_occupation(r, lad)) for r in rs])
    diffs = (S[1:] - S[:-1])
    check("entropy monotone non-decreasing in r", (diffs >= -1e-6).all().item(),
          f"min step={diffs.min().item():.2e}")


def test_unitary():
    blk = e.EntangledBlock(16, k=8, basis="eigen")
    _, _, UA, UB = blk.joint_state(torch.tensor([1.0]))
    I = torch.eye(8, dtype=torch.cfloat)
    ea = (UA @ UA.conj().T - I).abs().max().item()
    eb = (UB @ UB.conj().T - I).abs().max().item()
    check("U_A unitary", ea < 1e-4, f"{ea:.1e}")
    check("U_B unitary", eb < 1e-4, f"{eb:.1e}")


def test_reduced_state_eigs():
    blk = e.EntangledBlock(16, k=8, basis="eigen")
    M, p, _, _ = blk.joint_state(torch.tensor([1.3]))
    rho = M[0] @ M[0].conj().T                      # reduced state of A
    eigs = torch.linalg.eigvalsh(rho).real
    dev = (torch.sort(eigs)[0] - torch.sort(p[0])[0]).abs().max().item()
    check("reduced-state eigenvalues == Schmidt occupations", dev < 1e-4, f"max dev={dev:.2e}")


def test_grad_flow():
    for basis in ("eigen", "generic"):
        blk = e.EntangledBlock(16, k=8, basis=basis)
        out = blk(torch.randn(4, 6, 16))
        out.square().mean().backward()
        g = torch.sqrt(sum(pp.grad.square().sum() for pp in blk.parameters() if pp.grad is not None))
        check(f"{basis}: finite grads", torch.isfinite(g).item() and torch.isfinite(out).all().item(),
              f"grad_norm={g:.3f}")


if __name__ == "__main__":
    test_bounded_norm()
    test_product_to_entangled()
    test_monotone_entropy()
    test_unitary()
    test_reduced_state_eigs()
    test_grad_flow()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        sys.exit(1)
    print("ALL ENTANGLEMENT CHECKS PASSED")
