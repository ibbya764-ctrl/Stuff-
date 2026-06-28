"""CP2Model -- the full continuous-hypothesis-field architecture, composed.

7. What this unifies, as one object. The pipeline:

    shared operator -> continuous field psi(s,x) -> adaptive information-geometry
    -> moving scale samples -> local coherent consolidation -> permanent
    operator growth.

This module wires every subsystem into one forward pass:

  operator.py      (spine, -1/2 pinned)          field.py       (psi(s,x))
  density.py       (rho landscape)               scale_dynamics (5.6 motion)
  anti_collapse    (5.7 repulsion)               curvature.py   (5.9)
  observation.py   (3, coherence telemetry)      resolution.py  (5.5 two knobs)
  branches.py      (low-rank identities)         consolidation  (5 crystallize)
  geometry.py      (5.8 learned metric)          memory.py      (5.10 renewal)
  collapse.py      (assembly Psi(x))

STATUS, stated plainly: this is the complete *structure*, runnable on CPU on toy
data. It is NOT trained and NOT validated -- the operator spectrum, encoder, and
readouts are structural stand-ins for objects the GPU/Groq/Scaffold substrate
must train, and 3's coherence claim is demonstrated as a mechanism, not tested
as a claim. Per the program firewall: nothing here evidences the physics; the
-1/2 pin and the two-spectral-targets rule are enforced in code (operator.py).
A green forward pass means "the pieces compose and each behaves as specified",
not "the architecture works".
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from ..anti_collapse import AntiCollapseController, collapse_diagnostic
from ..scale_dynamics import gue_repulsion
from .operator import PinnedDilationOperator, gue_initialized_operator
from .field import ContinuousHypothesisField
from .density import DensityField
from .curvature import field_curvature, importance as importance_fn
from .observation import coherent_state, unitary_observation, projective_readout, \
    l1_coherence, purity
from .resolution import AdaptiveResolutionController
from .branches import LowRankBranchIdentities
from .consolidation import Consolidation
from .geometry import ScaleSpaceMetric
from .memory import MemoryGeometry
from .collapse import assemble


@dataclass
class CP2Model:
    """The composed architecture. Build with `CP2Model.build(...)`."""

    operator: PinnedDilationOperator
    density: DensityField
    anti_collapse: AntiCollapseController
    resolution: AdaptiveResolutionController
    consolidation: Consolidation
    memory: MemoryGeometry
    n_scales: int = 8
    d_out: int = 16
    use_curvature: bool = True          # 5.9 density x curvature vs density-only
    move_steps: int = 40                # 5.6 sample-motion steps per forward
    seed: int = 0

    @classmethod
    def build(cls, n_modes=48, n_scales=8, d_out=16, seed=0, use_curvature=True):
        op = gue_initialized_operator(n_modes, seed=seed)
        anchors = np.linspace(0.0, 1.0, 41)
        metric = ScaleSpaceMetric(anchors=anchors, expansion=0.05)
        return cls(
            operator=op,
            density=DensityField(bandwidth=0.3),
            anti_collapse=AntiCollapseController(bandwidth=0.5),
            resolution=AdaptiveResolutionController(max_samples=2 * n_scales),
            consolidation=Consolidation(op),
            memory=MemoryGeometry(op, metric, reset_every=25),
            n_scales=n_scales, d_out=d_out, use_curvature=use_curvature, seed=seed,
        )

    # -- 5.6 sample motion: density attraction + GUE repulsion + noise ----

    def _move_samples(self, scales, fld, rng, alpha=1.0, beta=0.3, temp=0.05,
                      dt=2e-3):
        s = np.asarray(scales, dtype=float).copy()
        for _ in range(self.move_steps):
            occ = fld.mode_occupations(s)
            attract = alpha * self.density.log_grad(s, occ)
            repel = beta * gue_repulsion(s)           # GUE-targeted force law
            noise = np.sqrt(2 * temp * dt) * rng.standard_normal(s.size)
            s = s + (attract + repel) * dt + noise
            s = np.clip(s, 0.0, 1.0)
        return np.sort(s)

    # -- the forward pass -------------------------------------------------

    def forward(self, x: np.ndarray) -> dict:
        """Run the full pipeline on a toy input vector; return answer+telemetry."""
        rng = np.random.default_rng(self.seed)
        # 1-2. spine -> continuous field psi(s,x)
        fld = ContinuousHypothesisField.from_input(
            self.operator, x, d_out=self.d_out, seed=self.seed)
        scales = np.linspace(0.05, 0.95, self.n_scales)

        # 5.6 move the scale samples (attraction + GUE repulsion + noise)
        scales = self._move_samples(scales, fld, rng)
        spread = collapse_diagnostic(scales)

        # field read-outs at the settled samples
        occ = fld.mode_occupations(scales)
        feats = fld.sample(scales)
        feat_norm = np.linalg.norm(feats, axis=1)

        # 2 + 5.9 density landscape and importance (density x curvature)
        rho = self.density.density(scales, occ)
        liquidity = self.density.liquidity(scales, occ)
        curv = field_curvature(scales, feat_norm)
        imp = importance_fn(rho, curv, use_curvature=self.use_curvature)

        # 3 coherence-preserving observation (mechanism demo, SUBSTRATE-GATED)
        coeffs = feats @ rng.standard_normal(self.d_out)   # scalar per sample
        rho_state = coherent_state(coeffs)
        obs_unitary = unitary_observation(rho_state)
        obs_proj = projective_readout(rho_state)
        coherence = {
            "unitary_purity": purity(obs_unitary),
            "unitary_coherence": l1_coherence(obs_unitary),
            "projective_purity": purity(obs_proj),
            "projective_coherence": l1_coherence(obs_proj),
        }

        # 5.5 stuck-signal -> adaptive rank + (optional) finer sampling
        norm = np.linalg.norm(occ, axis=1, keepdims=True) + 1e-12
        overlap = (occ / norm) @ (occ / norm).T
        disagreement = 1.0 - (overlap.sum(1) - 1.0) / max(len(scales) - 1, 1)
        difficulty = 1.0 - liquidity                  # flat/sparse = harder here
        stuck = self.resolution.stuck_signal(difficulty, disagreement, density=rho / rho.max())
        branches = LowRankBranchIdentities.init(
            len(scales), r=6, d=self.d_out, seed=self.seed,
            max_rank=4)
        eff_rank = branches.effective_rank(stuck)

        # 5 consolidation: crystallize solid, coupled samples into the spine
        nu_dom = self.operator.nu[np.argmax(occ, axis=1)]
        con = self.consolidation.step(occ, liquidity, nu_dom)

        # 5.8 + 5.10 information -> geometry, plastic metric, gated renewal
        mem = self.memory.tick(scales, usage=imp)
        geo_warn = self.memory.metric.collapse_warning(scales)

        # assemble: Psi(x) from many locally-useful regions
        local_fit = np.abs(feats) / (np.abs(feats).max(axis=0, keepdims=True) + 1e-12)
        answer, coeff = assemble(feats, imp, local_fit)

        return {
            "answer": answer,
            "scales": scales,
            "telemetry": {
                "operator_r_tilde": self.operator.level_spacing(),
                "operator_modes": self.operator.nu.size,
                "spine_size": int(self.operator.consolidated.sum()),
                "sample_spread_var": spread["var"],
                "sample_collapsed": spread["collapsed"],
                "density_mean": float(rho.mean()),
                "stuck_mean": float(stuck.mean()),
                "eff_rank_mean": float(eff_rank.mean()),
                "coherence": coherence,
                "consolidation": con,
                "memory": mem,
                "geometry_warning": geo_warn,
            },
        }


if __name__ == "__main__":
    m = CP2Model.build(n_modes=48, n_scales=8, seed=0)
    out = m.forward(np.arange(12.0))
    t = out["telemetry"]
    print("CP2Model forward pass [STRUCTURE runs on CPU; NOT trained/validated]")
    print(f"  answer dim          = {out['answer'].shape}")
    print(f"  operator <r~>       = {t['operator_r_tilde']:.3f}  "
          f"modes={t['operator_modes']} spine={t['spine_size']}")
    print(f"  sample spread Var   = {t['sample_spread_var']:.4f} "
          f"collapsed={t['sample_collapsed']}")
    print(f"  density mean        = {t['density_mean']:.3f}  "
          f"stuck mean={t['stuck_mean']:.3f}  eff-rank={t['eff_rank_mean']:.2f}")
    print(f"  coherence (unitary) = purity {t['coherence']['unitary_purity']:.3f}, "
          f"coh {t['coherence']['unitary_coherence']:.2f}")
    print(f"  coherence (project) = purity {t['coherence']['projective_purity']:.3f}, "
          f"coh {t['coherence']['projective_coherence']:.2f}")
    print(f"  consolidation       = {t['consolidation']}")
    print(f"  memory              = {t['memory']}")
    assert np.isfinite(out["answer"]).all(), "forward pass produced non-finite output"
    print("  -> full pipeline composed and ran end-to-end.")
