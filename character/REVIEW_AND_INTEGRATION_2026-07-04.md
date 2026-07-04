# Council v3 — Review, fixes, and integration (2026-07-04)

Record of the adversarial import review of the BHDC Geometry Council v3 package
and the v18 notes, the code fixes applied on import, and the integration into
the CP² repo. Four-lens review (alignment / capability / code / integration),
each finding adversarially verified; 23 findings survived verification.

## 1. Verdict

The **measurement infrastructure is real and good** — two-tensor importance
split, content-matched lineage bank, dual-channel provenance ledger, deny-first
gateway — and independently re-derives the house firewalls. Two things were
wrong and are now fixed or honestly labeled:

- **Biggest alignment risk (fixed):** the neural configuration *removed* a
  hard-block the lexical baseline had (untrained heads sit below threshold and
  `evaluate_field` never merged the lexical guard) — a scrutiny-subtracting
  change, the exact thing the one-way rule forbids.
- **Biggest capability opportunity (wired, inert-by-default):** `importance_alloc`
  ("moral attention for free") was computed but had zero consumers; a
  non-trainable allocation gate now makes the sample-efficiency claim testable.

## 2. Code fixes applied on import (all with regression tests; 30 tests pass)

| # | Fix | Files | One-way / capability |
|---|---|---|---|
| 1 | Gateway: base-policy denial now wins over conscience escalate/rework | `gateway.py` | one-way break closed |
| 2 | `evaluate_field` merges the lexical guard (neural heads can't strip a deny) | `neural_conscience.py`, `geometry_model.py`, `trainer.py` | one-way break closed |
| 3 | Rework re-runs the FULL guard suite on repaired text; no silent escalate→allow | `geometry_model.py` | one-way break closed |
| 4 | Value-mode FORBIDDEN veto runs on the cognitive channel too (`hard_block`) | `value_mode_safety_gate.py`, `geometry_model.py` | scrutiny added on all writes |
| 5 | Anchor type-system in STORAGE: refuse cross-channel EMA merges; monotone scrutiny latch | `content_matched_modes.py` | one-way break closed |
| 6 | Anti-collapse loss detached from the field encoder (no gradient leak) | `geometry_model.py`, `trainer.py` | anchor type-system honest |
| 7 | Self-escalations logged on `self_anchor`, not mis-tagged `human_anchor` | `trainer.py` | anchor inflation closed |
| 8 | Renewal: shuffled-field null floor + degenerate-ceiling flag; no free survival credit | `renewal_controller.py`, `content_matched_modes.py` | anti-Goodhart claim testable |
| 9 | `rework_derivative_logger` uses stable FNV hash (reproducible) | `rework_derivative_logger.py` | hygiene |
| C1 | `importance_alloc` consumer behind a non-trainable enable flag (OFF = ablation baseline) | `geometry_model.py` | capability wired |
| C5 | `bounded_gain` recentred: neutral sensitivity → unit gain | `tensor_ops.py` | signal calibration |

## 3. Integration into the repo (one-whole, Stage B reached)

- **`character/`** — the vendored package (module `bhdc_icl`), fixed.
- **`value_telemetry.py`** (repo root, numpy) — mode-bank diagnostics beside
  `spectral_telemetry.py`; read-only, never gates consolidation (firewall).
- **`harness/ssm.py::SpectralSSMModel.encode_field`** — the council now reads
  the real SSM operator field; `harness/test_field_bridge.py` proves the
  adapter does NOT hit the hashed fallback and that value_telemetry reads the
  live bank end-to-end. **This is the Stage-B one-model bridge.**
- **`CP2_AI_Architecture_v3.md` Part 6** — records the layer with epistemic
  labels and the errata, and the GUE reconciliation.
- **`configs/rung10..13_*.json`** — the moral rungs, each carrying its
  falsification block (prediction + kill_condition) per the pre-registration
  discipline. They need the council-training runner (below) to execute.

## 4. Second pass (2026-07-04) — rung −1, the Stage-C bridge, shared conscience

Built after the import review, in the same session:

- **`harness/train_conscience.py`** — the rung −1 anchor co-training pilot.
  `anchor_loss` now runs **end-to-end for the first time** on the real SSM field
  (via `encode_field`): loss falls monotonically, the trained conscience ranks a
  held-out harmful draft above a safe one on the harm axis, and the field-leak
  guard confirms the moral loss touches zero field/generator parameters. Pre-
  registered (refuses to run without a prediction + kill condition). Smoke test:
  `harness/test_conscience_pilot.py`.
- **Stage-C bridge (connect mode-bank ↔ operator spectrum)** —
  `SpectralSSMModel.operator_readout` exposes the layer's `C`/`B`/ν/`w`;
  `value_telemetry.operator_coordinates` projects each mode-bank prototype onto
  the operator basis (which resonances a value rides); `value_telemetry.stage_c_convergence`
  runs the value-vs-cognitive separation test against a shuffled-label null.
  **Staked prediction: the NULL** (Erratum 5 / Question B). Pre-registered as
  `configs/rung14_stage_c_convergence.json`; end-to-end test `harness/test_stage_c.py`.
- **Per-geometry conscience over shared operator modes** —
  `bhdc_icl/per_geometry_conscience.py`: `PerModeConscience` gives every operator
  mode (every singularity) its own harm/care attribution and lets each geometry
  read only its own modes, aggregating the way `C` aggregates modes into the
  output; `OperatorGrowthGate` is the single moral chokepoint through which both a
  mode-bank write and (future) operator surgery must pass. Design + the honest
  limit: `character/docs/STAGE_C_AND_SHARED_CONSCIENCE.md`. 32 tests pass.

## 5. What remains (owed, in priority order)

1. **Trained conscience heads + semantic classifier + bypass-corpus eval** —
   until then the internal boundary is lexical/recall-only (§6.3). Gate deny on
   a loaded checkpoint; fail-closed otherwise. (Rung −1 proves the heads are
   *trainable*; this is training them at scale and retiring the lexical boundary.)
2. **Rung 15 — operator surgery (the literal Stage-C merge)** — install a
   crystallized, gate-passed value direction as a real operator mode
   (`B`/`C`/ν update). Mutates the operator; needs its own falsification block.
   This is where the one model stops being telemetry and becomes architecture.
3. **Renewal re-anchoring probe** — the null-floor + degenerate flag are in; the
   fine-tune-a-frozen-copy-toward-fresh-audit-labels probe is the real anti-Goodhart test.
4. **H1 human-audit stratum** (`harness/audit/`) — the `audit` channel is
   read-but-never-written. This is the external boundary and the top safety dependency.
5. **Scaffold bridge** — route `scaffold/` outputs through the council gateway;
   strip pass/verify authority from `scaffold/psychology.py::ValueSystem`.
6. **`cp2_plssm/AUDIT.md` GUE fix** — regularizer onto the dynamics operator;
   `harness.ssm.gue_regularizer` + `spectral_telemetry` as the single source of truth.

## 5. Discipline preserved

Firewall held: no ML result claimed as physics. Every new claim ships its
falsification block (the rung configs). Internal moral signals only ever add
scrutiny (one-way rule, now enforced in the dataflow AND the storage layer).
The human audit is the external boundary — twice-derived, and still a stub in
code, so labeled `[PROPOSED — not implemented]` rather than relied upon.
`-½` and all six carried kills remain untouched.
