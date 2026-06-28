"""1. The continuous hypothesis field psi(s, x). [SPECULATIVE]

The note's central reframing: the object is not N discrete branches but a
continuous field psi(s) over scale s, sampled at a handful of representative
scales. psi(s) is the *orbit* of the dilation operator (operator.py) -- not a
bolt-on, but what D was already producing, sampled deliberately.

This module holds the field as complex modal amplitudes evolved to each scale
sample, plus the read-out of the field value at a query position x. Scale
samples are mobile (their positions are moved by scale_dynamics, 5.6); this
class owns the field *values*, not the sample motion.

Numpy-only, runnable on toy data. The field has no learned content here (no
trained operator/encoder), so values are structural, not semantic -- the
semantic content needs the trained substrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operator import PinnedDilationOperator


@dataclass
class ContinuousHypothesisField:
    """psi(s, x): operator orbit sampled at scales `s`, projected to features.

    `base_amplitudes` (N complex) is the field at scale 0; dilating to scale
    s_i gives psi(s_i); a fixed readout matrix C (d x N) maps modal amplitudes
    to a d-dim feature the rest of the model consumes.
    """

    operator: PinnedDilationOperator
    base_amplitudes: np.ndarray         # (N,) complex
    readout: np.ndarray                 # (d, N) complex readout C

    @classmethod
    def from_input(cls, operator: PinnedDilationOperator, x: np.ndarray,
                   d_out: int = 16, seed: int = 0) -> "ContinuousHypothesisField":
        """Encode a toy input vector x into base modal amplitudes.

        Structural encoder (random fixed projection) -- stands in for the
        trained B/encoder of the SSM. [SUBSTRATE-GATED: real encoder is trained.]
        """
        rng = np.random.default_rng(seed)
        N = operator.nu.size
        x = np.asarray(x, dtype=float).reshape(-1)
        B = rng.normal(size=(N, x.size)) / np.sqrt(x.size)
        base = (B @ x).astype(complex)
        C = (rng.normal(size=(d_out, N)) + 1j * rng.normal(size=(d_out, N))) / np.sqrt(N)
        return cls(operator=operator, base_amplitudes=base, readout=C)

    def amplitudes_at(self, scales: np.ndarray) -> np.ndarray:
        """Modal amplitudes psi-hat(s_i) at each scale. Shape (n_scales, N)."""
        scales = np.asarray(scales, dtype=float).reshape(-1)
        return np.stack([self.operator.dilate(self.base_amplitudes, s)
                         for s in scales], axis=0)

    def sample(self, scales: np.ndarray) -> np.ndarray:
        """Field features psi(s_i) = C psi-hat(s_i). Shape (n_scales, d)."""
        amps = self.amplitudes_at(scales)
        return amps @ self.readout.T

    def mode_occupations(self, scales: np.ndarray) -> np.ndarray:
        """Per-sample operator-mode occupation |psi-hat_k(s_i)|^2 (n_scales, N).

        This is the object the redundancy force (anti_collapse merge_or_repel)
        and consolidation read: which modes of the shared operator each sample
        occupies. Anchored to the operator spectrum, not a learned similarity.
        """
        amps = self.amplitudes_at(scales)
        occ = np.abs(amps) ** 2
        return occ / (occ.sum(axis=1, keepdims=True) + 1e-12)


if __name__ == "__main__":
    from .operator import gue_initialized_operator
    op = gue_initialized_operator(32, seed=1)
    fld = ContinuousHypothesisField.from_input(op, np.arange(8.0), d_out=16)
    scales = np.linspace(0.0, 1.0, 6)
    feats = fld.sample(scales)
    occ = fld.mode_occupations(scales)
    print("field self-check")
    print(f"  psi(s) features shape = {feats.shape}  (n_scales, d)")
    print(f"  occupations shape     = {occ.shape}    rows sum to "
          f"{np.round(occ.sum(1), 3).tolist()}")
    # dilation decays amplitude uniformly (the pinned width): later scales smaller
    amp_norms = np.linalg.norm(fld.amplitudes_at(scales), axis=1)
    print(f"  |psi-hat(s)| decays  = {np.round(amp_norms, 3).tolist()} "
          f"(uniform exp(-s/2) decay -> rigidity)")
    assert amp_norms[0] > amp_norms[-1]
