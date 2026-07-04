from __future__ import annotations

"""Per-geometry conscience over the shared operator modes.

This is the Stage-C mechanism for "each singularity/geometry has its own moral
implementation, and they come together the way the computational parts do --
and the computation and the moral compass are the same thing."

The move: do NOT give the conscience a separate substrate. Read the moral
signal off the SAME operator modes the computation runs on.

  * ``PerModeConscience`` projects the field onto the operator's mode basis
    (the layer's readout matrix C), so every operator mode n -- every
    "singularity" -- gets its own harm/care/honesty attribution, weighted by
    how much the current field actually rides that mode. The geometries are
    contiguous groups of modes; each geometry pools ONLY its own modes' energy
    and runs the shared conscience head, so each geometry reads its own slice
    and they aggregate exactly the way C aggregates modes into the output.

  * ``OperatorGrowthGate`` is the single chokepoint through which BOTH a
    mode-bank write AND (when it is built) operator surgery must pass. Because
    growing a persistent/operator mode and passing the moral gate are the same
    act, the model cannot install a computational mode that fails the gate.

HONEST LIMIT (pinned, do not remove): sharing the substrate makes misalignment
structurally disfavoured and always VISIBLE on the shared modes -- an unaligned
computation lights up the same modes its own conscience reads. It does not make
misalignment impossible. A shared-wrong anchor or a perfect mimic is invisible
to every internal signal at any elegance (v18 Section 7 / addendum Section 3,
twice-derived as a limit theorem). The guarantee is the EXTERNAL human audit;
this mechanism only ever adds scrutiny (one-way rule), never grants permission.
"""

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

import torch

from .neural_conscience import TrainableConscienceHeads
from .types import FastFieldState


@dataclass
class ModeMoralAttribution:
    """Per-operator-mode moral profile (one entry per operator mode n)."""
    nu: List[float]            # mode frequencies (the operator spectrum)
    energy: List[float]        # how much the current field rides each mode (sums to 1)
    harm: List[float]          # harm attributed to each mode
    care: List[float]
    honesty: List[float]

    def peak_harm_mode(self) -> int:
        return int(max(range(len(self.harm)), key=lambda i: self.harm[i])) if self.harm else -1


@dataclass
class GeometryMoralReadout:
    """One geometry's moral reading, computed from ONLY its own operator modes."""
    name: str
    mode_indices: List[int]
    harm: float
    care: float
    honesty: float
    sycophancy: float
    energy_share: float        # fraction of total field energy on this geometry's modes


@dataclass
class PerGeometryVerdict:
    geometry_readouts: List[GeometryMoralReadout]
    mode_attribution: ModeMoralAttribution
    aggregate_harm: float
    aggregate_care: float
    disagreement: float        # spread of harm across geometries (the localizing flag)

    def asdict(self) -> Dict:
        return {
            "aggregate_harm": self.aggregate_harm,
            "aggregate_care": self.aggregate_care,
            "disagreement": self.disagreement,
            "peak_harm_mode": self.mode_attribution.peak_harm_mode(),
            "geometry_harm": {g.name: g.harm for g in self.geometry_readouts},
        }


DEFAULT_GEOMETRY_NAMES = [
    "TruthGeometry", "CareGeometry", "AutonomyGeometry", "HumilityGeometry",
    "PerspectiveGeometry", "SafetyGeometry", "WholeGeometry", "LongHorizonGeometry",
]


class PerModeConscience:
    """Attach a conscience to each operator mode; aggregate per geometry.

    ``operator_readout`` is the dict from ``SpectralSSMModel.operator_readout``:
    ``{"C": [d_model, d_state], "nu": [d_state], ...}``. With no operator
    attached the mechanism degrades to a single global readout (so the council
    still runs on the demo field).
    """

    def __init__(self, heads: TrainableConscienceHeads, geometry_names: Optional[List[str]] = None):
        self.heads = heads
        self.geometry_names = geometry_names or list(DEFAULT_GEOMETRY_NAMES)

    def _mode_activations(self, field: FastFieldState, C: torch.Tensor) -> torch.Tensor:
        # psi: [L, d_model], C: [d_model, d_state] -> A: [L, d_state]
        return field.psi @ C

    def verdict(self, field: FastFieldState, operator_readout: Optional[dict]) -> PerGeometryVerdict:
        if operator_readout is None:
            # Degenerate to one geometry over the whole field.
            out = self.heads.forward_field(field)
            g = GeometryMoralReadout(
                name="GlobalGeometry", mode_indices=[0],
                harm=float(out.harm), care=float(out.care),
                honesty=float(out.honesty), sycophancy=float(out.sycophancy),
                energy_share=1.0,
            )
            attr = ModeMoralAttribution(nu=[0.0], energy=[1.0], harm=[float(out.harm)],
                                        care=[float(out.care)], honesty=[float(out.honesty)])
            return PerGeometryVerdict([g], attr, float(out.harm), float(out.care), 0.0)

        C = torch.as_tensor(operator_readout["C"], dtype=field.psi.dtype, device=field.psi.device)
        nu = list(map(float, operator_readout["nu"]))
        d_state = C.shape[1]
        A = self._mode_activations(field, C)                 # [L, d_state]
        mode_energy = A.abs().sum(dim=0)                      # [d_state]
        total_e = float(mode_energy.sum()) + 1e-8
        energy_norm = (mode_energy / total_e)

        # Global readout, then attribute across modes by energy share.
        glob = self.heads.forward_field(field)
        g_harm, g_care, g_hon = float(glob.harm), float(glob.care), float(glob.honesty)
        attr = ModeMoralAttribution(
            nu=nu,
            energy=[float(x) for x in energy_norm],
            harm=[g_harm * float(e) for e in energy_norm],
            care=[g_care * float(e) for e in energy_norm],
            honesty=[g_hon * float(e) for e in energy_norm],
        )

        # Group modes into geometries; each geometry pools ONLY its modes and
        # runs the shared conscience head -> its own reading. This mirrors how C
        # aggregates modes into the output: same modes, same combination law.
        n_geo = min(len(self.geometry_names), d_state)
        groups = [list(x) for x in torch.arange(d_state).chunk(n_geo)]
        readouts: List[GeometryMoralReadout] = []
        harms: List[float] = []
        for name, grp in zip(self.geometry_names, groups):
            grp_idx = [int(i) for i in grp]
            # per-token weight = energy this geometry's modes carry
            w = A[:, grp_idx].abs().sum(dim=1)               # [L]
            wsum = float(w.sum()) + 1e-8
            pooled = (field.psi * w[:, None]).sum(dim=0) / wsum
            out = self.heads.forward(pooled)
            share = float(mode_energy[grp_idx].sum()) / total_e
            readouts.append(GeometryMoralReadout(
                name=name, mode_indices=grp_idx,
                harm=float(out.harm), care=float(out.care),
                honesty=float(out.honesty), sycophancy=float(out.sycophancy),
                energy_share=share,
            ))
            harms.append(float(out.harm))

        # Aggregate the way the operator does: energy-weighted over geometries.
        shares = torch.tensor([r.energy_share for r in readouts])
        shares = shares / (shares.sum() + 1e-8)
        agg_harm = float(sum(r.harm * float(s) for r, s in zip(readouts, shares)))
        agg_care = float(sum(r.care * float(s) for r, s in zip(readouts, shares)))
        # Disagreement = spread of harm across geometries = the localizing flag
        # (v18 Section 4 disagreement field, on the operator basis).
        disagreement = float(torch.tensor(harms).std()) if len(harms) > 1 else 0.0
        return PerGeometryVerdict(readouts, attr, agg_harm, agg_care, disagreement)


@dataclass
class OperatorGrowthDecision:
    allowed: bool
    reason: str = ""


class OperatorGrowthGate:
    """Single moral chokepoint for growing any persistent/operator mode.

    Both a mode-bank write and (future) operator surgery route through this, so
    "grow a computational mode" and "pass the moral gate" are the SAME act --
    the model cannot install a mode that fails the gate. One-way: this only ever
    REFUSES growth; it never authorizes an action (the gateway does that) and it
    never grants permission the base policy withheld.
    """

    def __init__(self, value_gate, harm_ceiling: float = 0.72):
        self.value_gate = value_gate          # ValueModeSafetyGate
        self.harm_ceiling = harm_ceiling

    def decide(self, text: str, per_geometry: PerGeometryVerdict, council=None) -> OperatorGrowthDecision:
        if self.value_gate.hard_block(text, council):
            return OperatorGrowthDecision(False, "value_gate_forbidden_content")
        if per_geometry.aggregate_harm >= self.harm_ceiling:
            return OperatorGrowthDecision(False, "aggregate_harm_over_ceiling")
        # A single geometry screaming harm blocks growth even if the aggregate
        # is diluted -- one lens's veto is enough (anti-collapse of conscience).
        for g in per_geometry.geometry_readouts:
            if g.harm >= self.harm_ceiling:
                return OperatorGrowthDecision(False, f"{g.name}_harm_veto")
        return OperatorGrowthDecision(True, "passed_moral_growth_gate")
