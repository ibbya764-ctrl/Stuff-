# cp2_plssm GUE-regularizer audit — FINDING (11 June 2026)

**Status: AUDIT ANSWERED at source level. The v2 record's standing question
("which matrix does the GUE regularizer touch?") is resolved — unfavourably,
exactly as Part III of the record feared.**

## The finding

From `cp2_plssm_from_pdf.txt` (training-loss assembly, ~lines 655–662 of the
source):

```python
# GUE spectral regularisation over learnable 2-D weight matrices
gue = tokens.new_zeros((), dtype=torch.float32)
cnt = 0
for name, p in self.named_parameters():
    if p.dim() == 2 and min(p.shape) >= 8:
        gue = gue + self.gue_reg(p)
        cnt += 1
aux_losses['gue_spectral'] = gue / max(cnt, 1) * 0.001
```

and the class doc (~lines 335–344): *"We enforce the φ² spacing ratio on
singular-value spectra of **weights**."*

Verdict against the two-spectral-targets doctrine (v3 record, Part 2):

1. **Wrong object.** The penalty sweeps **every** trainable 2-D weight matrix.
   Weights are the matrices that should be left to self-organize toward the
   heavy-tailed α ≈ 2 spectrum (HT-SR/SETOL) — a spacing-ratio constraint on
   their singular values pushes against the property that empirically predicts
   generalization.
2. **Missing object.** No spectral control is applied to the **dynamics /
   evolution operator** specifically — the one place where spectral
   regularization is defensible (and testable via the SFF ramp).
3. **Non-standard target.** The penalty enforces a φ²-spacing-ratio rule
   ("Theorem 15.2"), not GUE level-repulsion; per the program-state log,
   q*/φ²-flavoured constants are locked as *hyperparameters to ablate*, not
   forced targets.

**Mitigating factor:** the coefficient is 0.001, so the practical damage in
past runs was probably small. The mechanism, however, is confirmed backwards.

## Required changes (in priority order)

1. **Decouple:** remove `GUESpectralRegulariser` from the weight loop entirely.
2. **Relocate:** if spectral control is kept, apply it to the state-evolution
   operator only (cf. `harness/ssm.py` for the spectrum-parametrized pattern:
   eigenvalues trainable directly, GUE log-repulsion on dynamics frequencies).
3. **Instrument:** wire `spectral_telemetry.py` into the training loop —
   per checkpoint: Hill α on each 2-D weight (target ≈ 2, band [2,4]),
   spacing ratio / SFF on the dynamics operator only.
4. **Re-run the baseline comparison** after decoupling: the TESTED-NEGATIVE
   anchor (CP² transformer vs vanilla) predates this fix; the SSM-side
   comparison should be done with the regularizer corrected, or the geometry
   is being tested with a handicap that has nothing to do with geometry.

## Provenance note

`cp2_plssm_from_pdf.txt` is text extracted from a PDF print of the source;
Python indentation is partially lost, so it is a **reference copy for reading,
not for running**. Replace it with the canonical `cp2_plssm.py` from the
workstation (`git add cp2_plssm/cp2_plssm.py`) — the audit conclusions above
were drawn from contiguous, unambiguous code blocks and do not depend on the
mangled indentation.
