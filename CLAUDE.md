# CLAUDE.md — session handoff for the CP² AI program

Read this first. It is the bridge between sessions (web/remote and local
workstation) and encodes the program's working discipline.

## What this repository is

The AI-engineering track of Ibby's CP² Projective-Lorentzian research program.
The long-term goal is an architecture that could plausibly lead toward AGI,
pursued with hard empirical discipline. The physics/mathematics track lives in
separate documents (latest program-state log: 11 June 2026); the two tracks
share vocabulary, **not** mechanisms (the "two-problem firewall").

## Repo map

| Path | What |
|---|---|
| `CP2_AI_Architecture_v3.md` | **The current consolidated record.** Read before changing anything architectural. Supersedes the v2 PDF. |
| `spectral_telemetry.py` | Numpy-only diagnostics: ⟨r̃⟩, SFF+ramp, Hill α, equivariance gate. Self-test: `python3 spectral_telemetry.py`. |
| `harness/` | Matched-budget experiment harness (runner enforces pre-registration), spectrum-parametrized SSM science arm, rung-1 optimizer test, rung-6 Scaffold-vs-plain eval. |
| `configs/` | Pre-registered experiment configs (rungs 2–4 A/B pairs + smoke; **rungs 10–13** = the moral rungs, each carrying its falsification block). |
| `scaffold/` | Scaffold/"Bri" — the ~87-file cognitive orchestration layer (epistemic pipeline, DMN daemon, memory, continual training, router...). Imported 11 June 2026; **cognitive benefits still unmeasured** (rung 6). |
| `cp2_plssm/` | `cp2_plssm_from_pdf.txt` (read-only reference; indentation mangled — replace with canonical `cp2_plssm.py` from the workstation) + `AUDIT.md` (**the regularizer audit finding — read it**). |
| `character/` | The BHDC Geometry Council — the program's **alignment layer** (module `bhdc_icl`). Vendored + reviewed + fixed 4 July 2026. Read `character/REVIEW_AND_INTEGRATION_2026-07-04.md` first; v18 theory notes in `character/docs/`. Tests: `PYTHONPATH=character python3 character/tests/run_all.py` (30 pass). |
| `value_telemetry.py` | Numpy mode-bank diagnostics (lineage, anchor-coord spread, renewal survival, drift) beside `spectral_telemetry.py`. Read-only; never gates. Self-test: `python3 value_telemetry.py`. |

## Current state (as of 4 July 2026)

1. **Audit ANSWERED:** cp2_plssm's GUE regularizer acts on **all 2-D weight
   matrices** (wrong object) and not on the dynamics operator (missing
   object). See `cp2_plssm/AUDIT.md` for the required fix. Coefficient was
   0.001, so past damage likely small. **Still owed** (Part 6.4).
2. **Harness ready:** all entry points smoke-tested on CPU. Real runs need the
   workstation GPU (rungs 2–4) or a Groq key / Scaffold endpoint (rung 6).
3. **Citations verified:** SETOL (arXiv:2507.17912, α→2), L-CNN 2024
   (arXiv:2401.06481), Muon, hybrid-SSM evidence (arXiv:2406.07887). No
   outstanding [VERIFY] flags in the v3 record.
4. **Character layer imported (4 July 2026):** BHDC Geometry Council vendored to
   `character/`, adversarially reviewed (23 verified findings), 11 code fixes
   applied (one-way-rule breaks closed; anchor type-system moved into storage;
   renewal null-floor; `importance_alloc` consumer wired inert-by-default), and
   the **Stage-B field bridge** (`SpectralSSMModel.encode_field`) proven: the
   council reads the real SSM field, not the demo. 30 tests + field-bridge tests
   pass. See `character/REVIEW_AND_INTEGRATION_2026-07-04.md` and v3 **Part 6**.
   The internal safety boundary is still **lexical/recall-only** — the trained-
   head + bypass-corpus boundary is owed (Part 6.3). The one-model identity is a
   **three-stage convergence**, Stage C (shared operator) unbuilt.

## Character layer state (4 July 2026 — second pass)

Built and green (CPU): rung −1 pilot (`harness/train_conscience.py`), the Stage-C
operator bridge + per-geometry conscience + `OperatorGrowthGate`, H1 audit stratum
**wired into the live renewal cycle**, rung 15 operator surgery, and the rung 16
forward-path moral coupling. The one-model merge is now built on both the memory
side (surgery) and the compute side (coupling). **Run the character rungs:**
`PYTHONPATH=.:character python -m harness.run_character_rung --all` (smoke) or
`... configs/rung1N_*.json --full` (GPU). On smoke: rung 15/16 PASS, rung 14 is
`UNDECIDED_insufficient_modes` (needs a full-scale mode bank). See
`character/docs/STAGE_C_AND_SHARED_CONSCIENCE.md`. Honest limit still pinned: the
fusion makes misalignment disfavoured + loud, not impossible; the human audit is
the guarantee.

## Next actions (priority order — from v3 Part 5.2 + Part 6.5)

1. **Run rungs 14–16 at scale** on the GPU with a trained conscience + a grafted
   instruction-tuned generator (the byte tokenizer produces text no judge can
   appraise). rung 14 needs a full-scale mode bank to leave UNDECIDED.
2. **Trained conscience heads + semantic classifier + bypass-corpus eval;**
   gate deny on a loaded checkpoint (fail-closed) — retire the lexical boundary.
3. **Renewal re-anchoring probe** + supply live H1 judgments to the renewal
   `audit_provider` (the wiring is in; it needs a real judgment feed).
4. **Start episodic logging in Scaffold now** (data accrues; needed for rung 7).
5. **Apply the AUDIT.md fix** to cp2_plssm; wire in `spectral_telemetry`.
6. **Rung 6:** `python3 -m harness.eval_rung6 --engine plain ...` vs
   `--engine http --url <scaffold>` at identical `--max-tokens`.
7. **Moral rungs 10–13** (`configs/`, pre-registered): need actions 1–2 first.
8. **Rung 1:** `python3 -m harness.optimizers`. **Rungs 2–4:** config pairs,
   3 seeds each, then `harness.runner.compare_runs(...)`.

## Non-negotiable discipline (from the program records)

- **Pre-registration:** the runner refuses configs without a written
  prediction + kill condition. Don't work around this; it is the point.
- **Matched budget:** never compare runs whose params/tokens differ >2%.
- **Two spectral targets:** GUE/level-repulsion → dynamics operator ONLY
  (diagnostic: SFF ramp). Weights → leave to self-organize; read Hill α ≈ 2.
  Never GUE-regularize weights.
- **Negatives are first-class.** A clean loss is a finding; record it in the
  v3 record with a [TESTED-NEGATIVE] label, don't soften it.
- **Two-problem firewall.** No claim that an ML result evidences the physics
  or vice versa. φ/q*-flavoured constants are hyperparameters to ablate.
- **Epistemic labels** ([ESTABLISHED]/[IMPLEMENTED]/[TESTED-NEGATIVE]/
  [PROPOSED]/[SPECULATIVE — gate hard]) are load-bearing. Preserve them.

## Environment notes

- Workstation: Lenovo ThinkStation, RTX 4060 Ti 16 GB, Ubuntu; Modal (A100)
  for burst. Size experiments accordingly (≤1B params, LoRA beyond).
- Scaffold serves via Cloudflare Tunnel with Groq as the engine
  (`GROQ_API_KEY`). No secrets are committed to this repo — keep it that way.
- Web/remote sessions see only this GitHub repo; the workstation filesystem
  is not visible to them. Anything that must survive a remote session has to
  be committed and pushed.
