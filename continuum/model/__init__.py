"""The composed continuous-hypothesis-field architecture (note §7 pipeline).

[STRUCTURE COMPLETE, NOT TRAINED/VALIDATED.] Every module §0–§5.10 of
`bhdc_continuous_branches_note`, wired into one `CP2Model` that runs a forward
pass on CPU. Real numpy implementations where a mechanism is CPU-expressible;
substrate-gated stand-ins (encoder/readout/operator spectrum, and §3's coherence
claim) where the GPU/Groq/Scaffold substrate is required.

Disciplines enforced in code, not just prose:
  * −½ pin: operator.CRITICAL_LINE; every consolidation asserts Re(λ)=−1/2.
  * Two spectral targets: GUE/level-repulsion measured on the operator
    frequencies only (operator.level_spacing); never on weights.
  * Two-problem firewall: no φ/q* constants; the log-gas/GUE machinery is used
    as an engineering target with a measurable signature.

See README.md for the module map and the honest scope statement.
"""

from .model import CP2Model
from .operator import PinnedDilationOperator, gue_initialized_operator, CRITICAL_LINE

__all__ = ["CP2Model", "PinnedDilationOperator", "gue_initialized_operator",
           "CRITICAL_LINE"]
