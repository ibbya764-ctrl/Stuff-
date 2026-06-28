"""Continuous-hypothesis-field arm of the CP2 AI program.

Built from `bhdc_continuous_branches_note` ([SPECULATIVE] throughout), one
arrow at a time, each validated against the last. This package holds ONLY the
pieces that are (a) flagged buildable-first by the note and (b) fully
validatable on CPU, numpy-only, without the GPU / Groq / Scaffold substrate:

  * anti_collapse  -- 5.7 AntiCollapseController. [ENGINEERING] The one
                      anti-collapse primitive (diversity loss / load-balancing
                      / sample repulsion are the same mechanism) made
                      first-class, plus the spread/variance collapse
                      diagnostic implemented once.
  * scale_dynamics -- 5.6 force-driven scale-sample motion: attraction +
                      repulsion + noise. The GUE-targeted (Dyson log-gas)
                      force law vs the generic Gaussian-kernel (SVGD) baseline
                      vs a no-repulsion collapse control.
  * eval_force_law -- the pre-registered A/B harness that scores the three
                      arms with Var(s) and the level-spacing <r~> statistic
                      (reusing spectral_telemetry), and reports SUPPORT /
                      FALSIFY against the note's 5.6 falsification block.

Discipline carried in from CLAUDE.md and the note:
  * Two-problem firewall: nothing here claims an ML result evidences the
    physics or vice-versa. phi/q* are absent; the only "physics" is the
    log-gas force law, used as an engineering target with a measurable
    signature, A/B'd against a generic baseline so it is tested not assumed.
  * The -1/2 pin is untouched: none of this code goes near the dynamics
    operator's real part. These are sample *positions* in scale-space.
  * This validates the MECHANISM/spacing half of 5.6 only. "Downstream task
    performance" needs the built model and is out of scope here -- logged, not
    silently dropped.
"""

__all__ = ["anti_collapse", "scale_dynamics", "eval_force_law"]
