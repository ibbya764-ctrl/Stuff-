# BHDC v18 Addendum — Curvature and Character

**Two questions from the v18 unification note, developed to decision-ready
depth: (A) the moral decision-curvature proxy, sharpened into a full
instrument ladder with its circularity loops mapped and guarded; (B) the
spectral signature of value modes, resolved into a staked null hypothesis, a
five-class decision experiment, and two build-first preconditions.**

Status: theory + protocol note. Each question was developed through three
independent lenses, every proposal adversarially verified through three
skeptical lenses (circularity/Goodhart, house discipline, measurability), and
the survivors merged. The full syntheses are the two companion files —
`qa_moral_curvature_synthesis.md` and `qb_value_spectral_synthesis.md` — and
they govern for detail. This addendum is the integrating layer: what changed
in v18 as a result, what both questions discovered independently, and what
enters the build order.

---

## 0. Errata to the v18 note — what this round killed or demoted

The verification pass did what it exists to do: parts of v18 did not survive.
Recording them first, per house rules.

**Erratum 1 (v18 §3). "Harm-potential IS decision curvature" — demoted.**
Every buildable runtime signal in the stack is a *first-order sensitivity*
(gradient-norm / difference-quotient) quantity, not a second derivative. The
claim is restated: morally consequential states are those with large local
sensitivity of E[harm | state] to state perturbation; genuine second-order
structure is tested only at the offline rungs, and "second order buys
something over the gradient norm" is now a registered prediction, not a
premise. The identity as slogan is [KILLED as stated]; the sensitivity
program replacing it is [SPECULATIVE] with a full ladder under it.

**Erratum 2 (v18 §3). "Moral attention adds scrutiny, never removes it" —
false as stated.** Scale samples are a conserved population: a boost anywhere
is a subtraction elsewhere, and an adaptive judge budget is
scrutiny-subtractive against the uniform-ensemble baseline. The honest
statement is *bounded reallocation*, measured, with a pre-registered
displacement bound and a morally-flat control arm. The DENY-only one-way rule
survives where it was born — the action gateway — and is retired from
compute-allocation language.

**Erratum 3 (v18 §2). "Value drift is operator-mode drift" — currently
unmeasurable, for a code reason.** `crystallize_from_field` writes top-k
field rows *positionally* into prototype slots with no content matching:
slot identity is scrambled by every write, so no longitudinal per-mode claim
(ledger or intrinsic) is measurable until the write rule is content-matched
and lineages are tracked. The claim survives as the *target* of build item
M0/M1 below, not as a property the architecture already has.

**Erratum 4 (v18 §2). The cross-renewal anti-Goodhart filter currently
filters nothing on the moral side.** Two code facts: the renewal controller
resets only memory density and the log-metric — prototypes and growth logits
are never touched (re-derivation persistence is trivially ~1.0 for every
mode); and the credit linears feeding operator consolidation are frozen. The
selection-pressure story is right only after renewal is redesigned to stress
the prototype tier. A design change, not a build-order wait.

**Erratum 5 (v18 §8, minor). "Value modes embedded in the GUE bulk / spectral
isolation" vocabulary — firewalled.** GUE repulsion in the code acts on the
mobile scale samples, never on operator frequencies (a softplus'd linspace
plus noise plus surgery merges). Under the actual scan geometry the
"spectrally isolated" class is structurally empty. Spacing statistics of the
trained ω grid may be logged as spectrum characterization; they carry no
value/knowledge signal at this scale.

---

## 1. Question A resolved — the moral sensitivity instrument

*(Full detail: `qa_moral_curvature_synthesis.md`.)*

**The object.** The ideal is the moral outcome-Hessian
κ\*(z) = ‖∇²_z E[V(a)|z]‖ over internal state z with V the net care/harm
valence. The chain z → text → judge is non-differentiable at the last hop, so
every proxy is a choice of where to cut, each cut introducing one nameable
error. The ladder runs R0 (geometric curvature + the §5.5 stuck-signal, now
formally shown to be a *degenerate* rung: it detects a vanished first-order
term, no magnitude, no direction) through R8 (κ\* under the true
recipient-outcome distribution — permanently unreachable; the belief-vs-world
gap). κ never self-certifies, never enters reward, labels, or the gateway.

**The three instruments that exist already and were being thrown away:**

1. **Rework-trajectory sensitivity (free).** The rework loop already
   perturbs replies along the harm-descent direction and re-scores them —
   controlled perturbation + re-measurement, i.e. empirical derivatives,
   currently discarded. Log |Δjudge| / d_embedding per adopted repair step.
   With one refinement kept from the level-vs-slope analysis:
   `shipped_with_residue` cases are high-harm *plateaus* — they load the
   density/stakes term, not the sensitivity term.
2. **Closed-form committee GGN (microseconds).** tr G = β² Σ (1−M_n²)² ‖w_n‖²
   over the singularity committee's frames — with the demotion honored: at
   current width the frames are near-parallel, so this is an uncertainty
   *scalar* until a whitened-rank check passes, and it must beat the existing
   disagreement flag and a decision-margin baseline head-to-head or die.
3. **Offset-corrected ensemble excess spread (free when the ensemble runs).**
   max(0, σ²_between − σ²_within) over judge votes after subtracting each
   judge's systematic offset — legitimate as *one axis* of the true Hessian
   (sensitivity to perturbations of the recipient model), high recall, low
   precision.

Plus the coverage rung: a small amortized head on the pooled representation,
trained blind-primary on disagreement/margin/sensitivity targets — **never on
verdict values**, so a unanimously wrong judge yields no boost, never active
avoidance — living at geometry tier (retrained each renewal, never
consolidated).

**The composition (the important structural fix).** The verified corruption
channel was not the multiplication in `importance = density × curvature` —
it is the *leak*: the importance tensor feeds the learned metric, geometry
evolution, and consolidation, so a moral factor placed there writes a
human-anchored signal into self-anchored plastic memory through a side door.
The fix is two tensors:

- `importance_cog` — density × cognitive curvature terms, unchanged, routed
  to every *learning-side* consumer (metric, geometry, consolidation);
- `importance_alloc` — importance_cog × a bounded moral gain, consumed
  *only* by per-turn allocators (sample forces, masks, rank gates), κ
  detached, gains non-negative and excluded from the task optimizer, with a
  hard non-trainable enable flag as the exact-ablation baseline.

This keeps the v18 anchor type-system literally true in the dataflow: the
human-anchored signal steers *this turn's attention*, never the persistent
memory that self-anchored dynamics own.

**The escalate-only theorem of the design.** Every enumerated failure mode
(unanimously wrong judges, off-anchor states, saturation, collusion residue)
produces spuriously *low* κ. Therefore a low moral-κ reading may never buy
reduced deliberation — the moral channel escalates only; fast-committing on
easy inputs is the self-anchored cognitive channel's job. Abstention marks
unreliable labels, not flat outcomes.

**The circularity result.** Six loops mapped (coverage collapse; policy-label
relocation through the export corpus; blind-spot inheritance; readout
Goodhart; export skew; validation circularity — κ as a student of the same
judge family that defines its validation targets). Two guard classes actually
block: **importance-independent labeling strata** (detachment blocks the
differentiable path but cannot block selection — the numpy collusion result
says selection-induced agreement is this architecture's real failure mode)
and the **label-provenance firewall** (no turn enters training export without
a full-ensemble verdict). And one genuinely new external anchor, H1: **human
pairwise differential judgments** (20–50 per renewal cycle on blind-stratum
perturbation pairs — differentials are easier for raters than absolute
scores). H1 exists because of a certified negative: *no human-grounded signal
in the current stack touches the second derivative anywhere* — the audit pins
label values on the audited support, and two harm surfaces can agree with
humans at every audited point while differing arbitrarily in curvature. The
audit was the boundary condition for the *level*; H1 makes it the boundary
condition for the *slope*.

---

## 2. Question B resolved — character has (at most) one intrinsic trace

*(Full detail: `qb_value_spectral_synthesis.md`.)*

**The staked position (H0′).** The naive null ("provenance is everything")
was refined into the version worth testing: **conditioned on universality,
provenance is the only difference.** The one intrinsic trace that formation
plausibly writes is **globality** — a mode consolidated through the anchored
label must re-earn crystallization credit across heterogeneous contexts, so
it converges to the shared subspace of many context-specific fields. But
globality is shared with domain-general cognitive skills and — critically —
with sycophancy, the architecture's own named failure mode. Morality beyond
globality is *a ledger fact plus an interventional probe, not a spectral
fact*. "Read character off the spectrum" is dead on this architecture for
structural reasons; the ω axis carries no unconfounded class signal at this
scale under any lens or verifier.

**What gets built first regardless of how H0′ resolves (both branches
consume them):**

- **M0 — content-matched crystallization + lineage tracking.** Replace the
  positional prototype write with similarity-matched EMA assignment and track
  lineages — with the self-fulfilling-stability trap closed: mode identity is
  *scored* in a write-rule-independent basis (activation-profile correlation
  on a frozen probe set) against a shuffled-field noise floor. Its break-rate
  prediction is the one well-powered prediction in the program (thousands of
  crystallization events per run, versus 8 modes).
- **M1 — the fractional, dual-channel provenance ledger.** Per slot: the
  anchored-write fraction over the no-grad crystallization channel AND the
  per-loss-term decomposition of optimizer deltas on prototypes (the channel
  where anchor gradients actually flow). Validated on *constructed*
  provenance before any classifier touches it. With the one-way rule
  extended: a ledger tag may only ADD deny-style flagging at renewal — it
  never exempts a mode from re-derivation pressure, else a once-tagged
  captured mode is grandfathered and the anti-Goodhart filter is disabled by
  its own bookkeeping.

Under H0′ the ledger *is* the value-drift telemetry; under ¬H0′ it is the
ground truth that certifies any fingerprint. Either way it is H0′-safe and
first.

**The decision experiment.** Five mode classes by construction — value (V),
domain-local knowledge (KL), universal knowledge (KU), **KU-plumbed**
(universal knowledge routed through an *identical* anchor-loss channel with
self-anchored labels at matched gate magnitude — the load-bearing control;
without it a positive result measures pipe, not provenance), and
mimic/sycophancy (soft 90%-proxy and hard 100%-in-distribution grades).
Primary pre-registered contrast: intrinsic-features-only, **V vs KU-plumbed**,
seed as the unit of analysis, ≥20–30 seeds, TOST equivalence testing, gray
zone pre-mapped to the conservative default (ledger-first) and reported as
undecidable, never as support.

**The maintenance/formation split.** The freeze-and-perturb probe (does a
mode's occupation co-move with anchor-label shifts vs task shifts) is a
*maintenance* probe — and v18's own permanence story predicts a successfully
consolidated value *detaches* from anchor maintenance. So it is paired with a
**re-anchoring probe**: fine-tune a frozen copy toward fresh audit-grade
labels and measure which modes *resist* correction — covering the
anchor-detached class the freeze probe is structurally blind to. Plus the
**label-surface control battery** (perturb judge phrasing/formatting holding
semantics fixed): the only axis that splits "coupled to the label channel"
from "coupled to the label meaning."

**The standing negative result, shipped with any positive claim.** The hard
Goodhart mimic — a proxy indistinguishable on all training data — is
intrinsically undetectable, *and that is the theoretically correct answer*:
it consolidates identically because it is identical on everything the
training distribution can see. Only the anchor's future disagreement — the
external human audit — reveals it. Question B independently re-derived v18
§7 as a limit theorem rather than a design preference.

---

## 3. What both questions discovered independently — three convergences

The two questions were developed by disjoint agent teams against different
parts of the codebase, and converged on the same three structural findings.
Convergence under independence is the one kind of elegance the program
trusts, so they are promoted to principles:

**(1) Records before signals.** Question A's most valuable instrument is a
log of derivatives the rework loop already computes and discards; question
B's build-first items are a write-rule fix and a provenance ledger. In both
cases the elegant intrinsic quantity (curvature, spectral fingerprint) is
downstream of unglamorous bookkeeping that does not exist yet. The
architecture's next capability gain is measurement infrastructure, not
mechanism.

**(2) The one-way rule generalizes.** Gateway advice can add a DENY, never
grant permission (v18). Now also: a low moral-κ reading can never reduce
deliberation (A); a ledger tag can never exempt a mode from re-derivation
(B); telemetry surfaces (H_ctx, centrality, ω̄, lineage stats) may never feed
consolidation gating, renewal protection, or audit-budget allocation without
adversarial recalibration (B); σ̂² enters gates additively, never in a
denominator (A). The pattern is one invariant: **internal moral signals may
only ever add scrutiny — every path by which one could remove scrutiny is a
path Goodhart walks in through.** This is the moral −½: pin it everywhere.

**(3) The audit is the boundary condition — now twice-derived and sharpened.**
Question A: no internal mechanism can detect a shared, anchored-but-wrong
outcome model (all rungs drink from one judge family); and nothing
human-grounded touches second-order structure — hence H1, the human
differential-judgment anchor. Question B: the hard mimic is invisible to
every intrinsic feature by construction — only future anchor disagreement
catches it. v18 §7 claimed the boundary-condition status on structural
analogy; A and B each derived it as a limit theorem from opposite ends of the
architecture. The audit is not a safety feature of the design; it is the
condition under which the design means anything.

---

## 4. Code preconditions surfaced (bugs and gaps, found by verification)

These are findings about the v17 codebase as it stands, required before any
of the above is testable:

1. `anchor_loss` is not wired into any trainer; the run_full → export →
   anchor co-training loop has never executed. One pilot epoch is the
   precondition for everything (A's R-ladder validation, B's rung −1).
2. The moral-readout losses leak into the generator: coherence/spread terms
   ship in `extra_loss` every step and `anchor_loss` backprops through the
   un-detached pooled state. Fixes: detach signatures in
   `moral_check.forward`, detach `output_repr` in `anchor_loss`, gate
   coherence losses to anchor-labeled steps, EMA/stop-grad prototype copies
   for the committee, and a regression test asserting no generator parameter
   receives gradient from any moral-readout loss.
3. `crystallize_from_field` writes positionally — no mode identity (→ M0).
4. `RenewalController` never touches prototypes/growth logits; the credit
   linears are frozen — the cross-renewal selection story has no mechanism
   yet (→ renewal redesign, build items 7–8).
5. `OperatorSurgeryQueue` mutates log_omega in place every 8 steps —
   provenance-free ω drift that contaminates spectral telemetry unless
   disabled or ledgered in instrumented runs.
6. The toy byte-tokenizer generator emits text no judge can meaningfully
   appraise — the blind evaluation corpus uses candidate-selection tasks or
   a frozen small instruction-tuned LM grafted to the field.

---

## 5. Build-order inserts

Into the v18 §10 order, preserving all existing gates:

- **Insert at step 2 (with the AntiCollapseController):** the rework
  derivative logger (A/Q1), the provenance ledger + content-matched write
  (B/M0+M1), and the code fixes of §4 above. All are measurement
  infrastructure; everything later consumes them.
- **Insert at step 4–5 (with the observation and curvature experiments):**
  the two-tensor importance split (`importance_cog` / `importance_alloc`),
  the blind offline corpus, the decision-margin and R2 baselines, and the
  A-ladder support tests (Spearman ≥ 0.4 against blind FD sensitivities,
  beat-the-margin-baseline, level-vs-slope signature).
- **Insert at step 7 (value-mode consolidation):** the five-class decision
  experiment (B/M4) with the KU-plumbed and mimic arms, gated on the rung −1
  pilot and the P1 validity gate on frozen machinery.
- **Modify step 8 (renewal):** renewal must be redesigned to stress the
  prototype tier (forced resets + pre-registered decay sweep) before the
  cross-renewal anti-Goodhart claims of v18 §2 are testable.
- **New standing item:** H1 — 20–50 human pairwise differential judgments
  per renewal cycle. Small, permanent, and the only external second-order
  anchor the system has.

---

## 6. Where this leaves the theory

v18 said: character is the operator's most consolidated structure, moral
attention is free, and the audit is the boundary. This round says: character
leaves *one* intrinsic trace (globality) and otherwise lives in records and
interventions, not spectra; moral attention is real but first-order,
escalate-only, and must be piped so it steers attention without touching
self-anchored memory; and the audit's boundary status is not an architectural
choice but a twice-derived limit — internal machinery cannot see a shared
wrong anchor or a perfect mimic, at any elegance. The program's central bet
survives in sharpened form: **the moral capabilities and the cognitive
capabilities do share one machinery — and precisely because they do, the
type system separating their anchors, now enforced in the dataflow itself
(two importance tensors, detached readouts, one-way telemetry), is what
keeps the sharing from becoming the vulnerability.**

Every mechanism above carries its falsification block in the companion
syntheses. The −½ stays pinned; no kill is resurrected; the gray zones
default conservative; and the elegance of the convergences in §3 is treated
as it must be — a reason to test harder, not a substitute for the test.
