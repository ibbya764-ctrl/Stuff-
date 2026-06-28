"""5.5 Two knobs of one density-driven resolution controller. [SPECULATIVE;
buildable pieces flagged [ENGINEERING-feasible].]

Branch count (sampling rate on psi) and branch distinctness (rank of each
branch's identity) both adapt to the input, from the SAME signal and in the
SAME direction: spend resolution where the model is stuck, save it where things
are settled.

The stuck-signal -- already in the architecture, read off quantities the model
already computes, no new measurement:

    high difficulty (high output entropy / high loss) AND low branch
    disagreement (the coherent superposition is nearly degenerate) = "I can't
    resolve this and my branches aren't giving me different angles."
    Where the field is also dense, the same conclusion holds.

  * Knob 1 -- adaptive rank (branches.py gate). [ENGINEERING-feasible]
  * Knob 2 -- adaptive local sampling: refine psi(s) finer in the stuck
    neighbourhood (no spawning -- finer sampling of one continuous landscape).

Both bounded by a HARD CAP and a "give up and commit" fallback (the mandatory
runaway caution): when refinement past the cap does not reduce the stuck-signal,
stop and collapse on the best current estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptiveResolutionController:
    """Computes the stuck-signal and the two resolution responses.

    Caps: `max_rank` (knob 1, in branches), `max_samples` (knob 2). The
    give-up fallback fires when extra resolution stops reducing stuckness.
    """

    difficulty_w: float = 1.0
    disagreement_w: float = 1.0
    density_w: float = 0.5
    max_samples: int = 32
    refine_thresh: float = 0.6          # stuck-signal above which to refine

    def stuck_signal(self, difficulty: np.ndarray, disagreement: np.ndarray,
                     density: np.ndarray | None = None) -> np.ndarray:
        """Per-sample stuck-signal in [0, 1].

        Rises with difficulty and with density, FALLS with branch disagreement
        (the conjunction: hard AND branches agree-too-much). All inputs assumed
        pre-normalized to ~[0,1].
        """
        d = self.difficulty_w * np.asarray(difficulty, float)
        a = self.disagreement_w * (1.0 - np.asarray(disagreement, float))
        sig = d * a
        if density is not None:
            sig = sig + self.density_w * np.asarray(density, float) * d
        # squash to [0,1]
        return 1.0 / (1.0 + np.exp(-4.0 * (sig - 0.5)))

    def refine_scales(self, scales: np.ndarray, stuck: np.ndarray) -> np.ndarray:
        """Knob 2: insert finer samples around stuck neighbourhoods.

        For each stuck sample, draw two nearby scales (s +/- small) -- finer
        sampling of the same continuous landscape, never splitting an object.
        Capped at max_samples (runaway guard). Returns the new scale set.
        """
        s = list(np.asarray(scales, dtype=float).reshape(-1))
        order = np.argsort(-stuck)                    # most stuck first
        for i in order:
            if len(s) >= self.max_samples:
                break                                 # HARD CAP -> commit
            if stuck[i] < self.refine_thresh:
                break
            s0 = scales[i]
            s.extend([s0 - 0.01, s0 + 0.01])
        return np.sort(np.array(s))

    def should_commit(self, stuck_before: float, stuck_after: float,
                      n_samples: int) -> bool:
        """Give-up-and-commit fallback: refinement past the cap, or no
        reduction in stuckness, => stop refining and collapse on best estimate.
        """
        no_progress = stuck_after >= stuck_before - 1e-3
        return bool(n_samples >= self.max_samples or no_progress)


if __name__ == "__main__":
    ctrl = AdaptiveResolutionController(max_samples=12)
    difficulty = np.array([0.9, 0.9, 0.2, 0.1])
    disagreement = np.array([0.1, 0.9, 0.1, 0.9])   # low disagree = stuck
    stuck = ctrl.stuck_signal(difficulty, disagreement)
    print("resolution self-check")
    print(f"  stuck-signal = {np.round(stuck, 3).tolist()}")
    # hard+agree (sample 0) is most stuck; easy+disagree (sample 3) least.
    assert stuck[0] == stuck.max() and stuck[3] == stuck.min()
    scales = np.array([0.0, 0.3, 0.6, 0.9])
    refined = ctrl.refine_scales(scales, stuck)
    print(f"  scales {scales.tolist()} -> {np.round(refined, 2).tolist()} "
          f"(finer where stuck)")
    assert refined.size > scales.size
    print(f"  commit? past-cap={ctrl.should_commit(0.8, 0.8, 12)} "
          f"(no-progress / cap -> collapse on best estimate)")
