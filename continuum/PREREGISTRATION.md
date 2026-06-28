# Pre-registration — §5.6 force-law spacing experiment (CPU mechanism test)

Pre-registered per the program discipline (the runner refuses configs without a
written prediction + kill condition; this arm runs outside the GPU runner, so
the registration lives here and is committed **before** the multi-seed harness
is run). Epistemic label: **[SPECULATIVE]** for the architectural claim;
**[ENGINEERING]** for the anti-collapse primitive it exercises.

## What is being tested

The **spacing half** of the note's §5.6 falsification block, in isolation from
the unbuilt model: *does force-driven dynamics with the GUE-targeted (Dyson
log-gas) repulsion actually produce GUE-like level statistics in the sample
positions, and is it distinguishable from the generic Gaussian-kernel (SVGD)
baseline and from no repulsion?*

Three arms, identical confinement (α=2, σ=1) and noise (T=1), 48 samples,
multiple seeds, `<r~>` measured on post-burn-in snapshots via
`spectral_telemetry.spacing_ratio`:

| arm    | repulsion law                              | role          |
|--------|--------------------------------------------|---------------|
| `gue`  | Dyson log-gas, F=Σ 1/(s_i−s_j)             | claim under test |
| `rbf`  | Gaussian/RBF kernel (the §5.7 primitive)   | A/B baseline  |
| `none` | none                                       | collapse control |

## Prediction (committed before running)

- **`gue` arm** → positions exhibit GUE level statistics:
  `<r~> ∈ [0.56, 0.63]` (target `R_GUE = 0.5996`), and `Var(s)` bounded away
  from 0 (≈ the semicircle value 24, structured spread — not collapsed).
- **`none` arm** → Poisson local statistics: `<r~> ∈ [0.36, 0.42]`
  (target `R_POISSON = 0.3863`); with low confinement noise it also contracts
  toward the attractor (collapse control behaves as the failure it guards).
- **`rbf` arm** → **distinguishable from `gue`**: `<r~>` falls *outside*
  `[0.56, 0.63]`. (Expected more rigid → higher `<r~>`, but the only committed
  claim is non-coincidence with the GUE band.)

## Kill condition / falsification (note §5.6 block, spacing half)

- If the **`gue` arm fails to reach the GUE band** (`<r~>` outside
  `[0.56, 0.63]` at matched sample count, multi-seed), the log-gas force law
  does **not** produce the claimed statistics even in the clean toy → the
  GUE-specific claim earns nothing where it should be easiest, and the generic
  repulsion (which still does the anti-collapse job) is the right default.
  **This falsifies the GUE-specific claim, not the repulsion itself.**
- If the **`rbf` arm also lands in the GUE band**, the generic kernel
  reproduces GUE spacing for free → again the GUE-specific choice earns
  nothing over the baseline (partial falsification, per the note: "Falsify:
  GUE-targeted repulsion matches or underperforms the generic kernel").

## Coverage limit (logged, not silently dropped)

This toy tests **only** the spacing/structured-spread half of the §5.6 block.
The **"downstream task performance"** half — whether GUE-targeted spacing yields
better task results at matched sample count — needs the built model and the
GPU/Groq/Scaffold substrate, and is **not** tested here. A green result here is
necessary, not sufficient: it says the mechanism produces the claimed
statistics, not that the statistics help the architecture.

## Firewall / pins

- **Two-problem firewall:** the log-gas is used as an engineering target with a
  measurable signature, A/B'd against a generic baseline. No claim that this
  result evidences the physics program, or vice-versa.
- **−½ pin untouched:** these are sample *positions* on the scale axis, never
  the dynamics operator's real part.
