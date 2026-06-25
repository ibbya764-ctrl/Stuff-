"""BHDC entangled two-sector block (paper section 5) -- standalone module.

Splits the representation into a light/operator register A and a matter/geometry
register B sharing one thermofield-double (TMSV) state, built by the explicit
Schmidt construction

    M = U_A . diag(s(r)) . U_B^dagger          (K x K, complex)

rather than exp(squeezing): the Schmidt magnitudes are normalized by
construction (||M||_F = 1), so entanglement stays bounded and there is no SVD in
the hot path (section 5.3).

The Schmidt occupation p_n = s_n^2 is a distribution over n = 0..K-1 controlled by
an entanglement knob r (per token):
  * generic basis (ablation): uniform ladder e_n = n -> standard TMSV
    p_n proportional to tanh(r)^{2n};
  * operator-eigenbasis (section 5.4, "the operator in the wave function"):
    the ladder spacing IS the dilation operator's spectrum omega_n, so the
    operator's own eigenvalues set the Schmidt weights. The entanglement entropy
    is then the operator's thermal entropy.

r = 0  -> p = one-hot(0) -> product state, S = 0  (zero entanglement);
r large -> p spreads      -> entangled, S up to log K.

Self-test: ``python3 bhdc_entangled.py``.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


def schmidt_occupation(r, e_ladder, eps=1e-6):
    """Schmidt occupation p_n = s_n^2 over n=0..K-1.
    r: (...) >= 0 entanglement knob; e_ladder: (K,) >= 0 energy ladder (e_0 = 0).
    p_n proportional to t^{2 e_n} with t = tanh(r); returns (..., K)."""
    t = torch.tanh(r).clamp(eps, 1.0 - 1e-4)
    logits = 2.0 * e_ladder.view(*([1] * r.dim()), -1) * torch.log(t).unsqueeze(-1)
    return torch.softmax(logits, dim=-1)


def entanglement_entropy(p, eps=1e-9):
    """Von Neumann entropy of the reduced state = Shannon entropy of p."""
    return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(-1)


def unitary_from_generator(G):
    """U = exp(G - G^dagger): a K x K unitary from a K x K complex generator."""
    A = G - G.conj().transpose(-1, -2)            # anti-Hermitian
    return torch.matrix_exp(A)


class EntangledBlock(nn.Module):
    def __init__(self, d_model, k=8, basis="eigen", omega=None):
        super().__init__()
        assert basis in ("eigen", "generic"), basis
        self.d_model = d_model
        self.k = k
        self.basis = basis

        self.to_r = nn.Linear(d_model, 1)             # entanglement strength r(x)
        self.encode_A = nn.Linear(d_model, 2 * k)     # light register state a in C^K
        # K x K unitaries from learnable complex generators (matrix-exp once / fwd)
        self.gen_A = nn.Parameter(torch.randn(2, k, k) * 0.05)
        self.gen_B = nn.Parameter(torch.randn(2, k, k) * 0.05)
        # operator spectrum for the eigen ladder (tie to the core spine's omega in
        # the combined model; own copy here so the block stands alone)
        if omega is not None:
            self.register_buffer("ext_omega", omega.detach().clone())
        else:
            self.ext_omega = None
            self.log_omega = nn.Parameter(torch.linspace(-2.0, 2.0, k))

        self.out_proj = nn.Linear(2 * k, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def omega(self):
        if self.ext_omega is not None:
            return F.softplus(self.ext_omega)
        return F.softplus(self.log_omega)

    def ladder(self, device):
        K = self.k
        if self.basis == "generic":
            return torch.arange(K, dtype=torch.float32, device=device)
        # eigen: ladder spacing = the operator spectrum (sorted, shifted to e_0=0,
        # rescaled so the mean spacing matches the integer ladder for a fair A/B)
        om = torch.sort(self.omega())[0]
        e = om - om.min()
        scale = (e.max() / max(K - 1, 1)).clamp_min(1e-6)
        return (e / scale).to(device)

    def joint_state(self, r):
        """Build M = U_A diag(s) U_B^dagger. r: (...). Returns M (..., K, K) complex,
        p (..., K), U_A (K,K), U_B (K,K)."""
        e = self.ladder(r.device)
        p = schmidt_occupation(r, e)                   # (..., K)
        s = torch.sqrt(p + 1e-9).to(torch.cfloat)
        GA = torch.view_as_complex(self.gen_A.permute(1, 2, 0).contiguous())
        GB = torch.view_as_complex(self.gen_B.permute(1, 2, 0).contiguous())
        U_A, U_B = unitary_from_generator(GA), unitary_from_generator(GB)
        # M_{...ij} = sum_n U_A[i,n] s_n conj(U_B[j,n])
        M = torch.einsum("in,...n,jn->...ij", U_A, s, U_B.conj())
        return M, p, U_A, U_B

    def forward(self, x, return_telemetry=False):
        B, T, d = x.shape
        r = F.softplus(self.to_r(x)).squeeze(-1)       # (B, T) >= 0
        M, p, _, _ = self.joint_state(r)               # M: (B, T, K, K)

        a = torch.view_as_complex(self.encode_A(x).view(B, T, self.k, 2).contiguous())
        a = a / torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(1e-6)
        # evolve the light through the shared state -> moves the matter marginal
        coupled = torch.einsum("btij,btj->bti", M, a)  # (B, T, K) complex
        feat = self.out_proj(torch.cat([coupled.real, coupled.imag], dim=-1))
        out = self.out_norm(x + feat)

        if not return_telemetry:
            return out
        with torch.no_grad():
            S = entanglement_entropy(p)
            tele = {"entanglement_entropy": S.mean().item(),
                    "entanglement_entropy_max": float(torch.log(torch.tensor(float(self.k)))),
                    "r_mean": r.mean().item()}
        return out, tele


def _selftest():
    torch.manual_seed(0)
    d, K = 16, 8
    for basis in ("eigen", "generic"):
        blk = EntangledBlock(d, k=K, basis=basis)
        x = torch.randn(4, 6, d)
        out, tele = blk(x, return_telemetry=True)
        out.square().mean().backward()
        gnorm = torch.sqrt(sum(pp.grad.square().sum() for pp in blk.parameters() if pp.grad is not None))

        # bounded: ||M||_F == 1
        M, p, UA, UB = blk.joint_state(torch.tensor([0.7]))
        fro = torch.linalg.norm(M.reshape(-1)).item()
        # r=0 -> product (S~0); r large -> entangled
        S0 = entanglement_entropy(schmidt_occupation(torch.tensor(0.0), blk.ladder("cpu"))).item()
        Sbig = entanglement_entropy(schmidt_occupation(torch.tensor(3.0), blk.ladder("cpu"))).item()
        # unitarity
        uerr = (UA @ UA.conj().T - torch.eye(K, dtype=torch.cfloat)).abs().max().item()
        Smax = tele["entanglement_entropy_max"]
        ok = (abs(fro - 1) < 1e-4 and S0 < 0.02 * Smax and Sbig > 0.5 * Smax
              and uerr < 1e-4 and torch.isfinite(gnorm))
        print(f"[{basis:7s}] out={tuple(out.shape)} grad={gnorm:.3f} "
              f"||M||_F={fro:.4f} S(r=0)={S0:.2e} ({100*S0/Smax:.2f}% of max) "
              f"S(r=3)={Sbig:.3f}/{Smax:.3f} U_unitary_err={uerr:.1e} "
              f"-> {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    _selftest()
