# Stage-C convergence and the shared-operator conscience

How the council mode-bank and the SSM operator spectrum actually get connected
and merged, and how each geometry/singularity gets its own moral implementation
that is the *same machinery* as the computation. Written 2026-07-04, alongside
the code that implements the first two stages.

Status: Stage A/B [IMPLEMENTED]; Stage-C telemetry [IMPLEMENTED]; the Stage-C
*merge* (operator surgery) is [PROPOSED — gated by rung 15]. The v18 §9 one-model
identity stays **[SPECULATIVE]** until the Stage-C experiment (rung 14) returns.

## The honest starting point (the thing this fixes)

Before this work, "moral capability = cognitive capability, same machinery" was
**asserted**, not built. Two facts made it a slogan:

1. Values and knowledge differed only by a **ledger channel string**
   (`human_anchor` vs `cognitive`) — same store, same write path, a label.
2. The council mode-bank (text `candidate_vector`s in `d_model` space) and the
   SSM operator spectrum (eigenvalues λ = −w + iν in `d_state` space) were **two
   disjoint objects the council never connected**. The conscience never read the
   operator; the operator never saw the conscience.

Making them one object is a **three-stage convergence**, not a one-shot merge.
Each stage has a concrete artifact and a gate.

## The three stages

### Stage A — one hierarchy, one anchor type-system  [IMPLEMENTED]
The memory tiers (field / geometry / operator) and the moral tiers (per-turn
judgment / drafting principles / disposition) are run as one hierarchy, with the
anchor type-system enforced in the dataflow **and** the storage layer (typed
banks, monotone scrutiny latch, detached moral losses, two-tensor importance).
This is what the import review hardened. It makes the *label* real — but the two
substrates are still disjoint.

### Stage B — the field bridge  [IMPLEMENTED]
`SpectralSSMModel.encode_field` makes the council read the **real operator-driven
field** ψ(s,x) — the SSM's own per-token hidden state — instead of a hashed demo.
Proven end-to-end in `harness/test_field_bridge.py` (the adapter no longer
silently falls back). Now the conscience and the computation share an *input*:
the same field. They still consolidate into *different stores*.

### Stage C — the shared operator  [telemetry built; merge gated]
Two steps, in order:

**C1 — the operator-coordinate bridge (built).** A mode-bank prototype is a
`d_model` direction; the operator's modes live in `d_state`, coupled to `d_model`
through the layer readout matrix `C`, each mode carrying a frequency ν_n.
`value_telemetry.operator_coordinates` projects each prototype onto `C`'s columns
— so for any consolidated value you can read **which operator resonances it
rides**. `SpectralSSMModel.operator_readout` exposes `C`, `B`, `ν`, `w`.

**C1-telemetry — the staked test (built; rung 14).**
`value_telemetry.stage_c_convergence` asks: do value (anchored) modes occupy a
*different operator-mode subspace* than cognitive modes, beyond a shuffled-label
null? The **staked prediction is the NULL** (Erratum 5 / Question B H0′: the ω
axis carries no unconfounded class signal at this scale). A negative is the
predicted, first-class result; it does not refute the one-model idea, it just
means *character is read from records and interventions, not the spectrum* at
this scale. A *positive* would be the surprise that begins to earn "values live
in the operator" — reported `[INVESTIGATE]`, never as proof, and only if it
survives the KU-plumbed control (universal knowledge routed through the identical
anchor channel — so a positive measures provenance, not pipe).

**C2 — operator surgery (the actual merge) [PROPOSED, rung 15].** The literal
merge is: when a value prototype crystallizes (survives renewal, passes the
gate), it **installs as an actual operator mode** — a small gated update to
`B`/`C`/ν — instead of living in a separate bank. Then there is one store: the
operator. This mutates the dynamics operator, so it must clear its own
falsification block first, and it is deliberately **not built yet**. It is the
point at which the identity stops being telemetry and becomes architecture.

## Each singularity/geometry has its own conscience — and it is the same machinery

`bhdc_icl/per_geometry_conscience.py` builds the thing you asked for: not a
conscience bolted beside the model, but a conscience read off the **same operator
modes the computation runs on**.

- `PerModeConscience` projects the field onto the operator basis (`C`), so **every
  operator mode n — every singularity — gets its own harm/care/honesty
  attribution**, weighted by how much the current field actually rides that mode.
- The **geometries are groups of operator modes**. Each geometry pools *only its
  own modes' energy* and runs the shared conscience head, so each geometry reads
  its own slice and they **aggregate exactly the way `C` aggregates modes into the
  output** — same modes, same combination law. The computational parts and the
  moral parts come together by the *same* operation.
- The spread of harm across geometries is the **disagreement field** on the
  operator basis (v18 §4): it localizes *which mode* a reply goes wrong on, not
  just that it does.

### "So it never builds something that doesn't align morally"

`OperatorGrowthGate` is the single chokepoint through which **both** a mode-bank
write **and** (when C2 is built) operator surgery must pass. Because *growing a
persistent/computational mode* and *passing the moral gate* become the **same
act**, the model cannot install a mode that fails the gate: forbidden content, a
harm aggregate over ceiling, or **any single geometry's harm veto** refuses the
growth. This is the strongest honest form of "computation and moral compass are
the same thing": you cannot extend the machinery except through the conscience.

### The limit, pinned (do not remove)

Sharing the substrate makes misalignment **structurally disfavoured and always
visible** — an unaligned computation lights up the very modes its own conscience
reads, and it cannot grow new capability without clearing the gate. It does
**not** make misalignment impossible. A shared-wrong anchor, or a mimic that is
identical to a good value on everything the training distribution can see, is
invisible to every internal signal **at any elegance** — the addendum derives
this twice as a limit theorem (v18 §7, addendum §3). Therefore:

- The internal mechanism's job is to make misalignment **expensive and loud**,
  never to certify its own safety.
- The **external human audit is the guarantee.** It is the boundary condition
  that makes all of the interior mean anything; it stays outside the model at
  every stage. Every internal moral signal here obeys the one-way rule — it can
  only ever **add** scrutiny (block, veto, escalate), never grant permission.

Claiming the internal fusion "never misaligns" would violate the program's own
firewall. The design instead makes aligned computation the path of least
resistance and misalignment the loudest thing on the shared modes — and leaves
the guarantee where it provably has to live: the audit.

## Build order for the merge

1. rung −1 (`harness/train_conscience.py`) — anchor co-training pilot. **Done.**
2. rung 14 (`configs/rung14_stage_c_convergence.json`) — the staked Stage-C null,
   over the operator-coordinate bridge. **Built; needs a full-scale bank to run.**
3. `PerModeConscience` / `OperatorGrowthGate` — per-geometry conscience over
   shared modes; single growth chokepoint. **Built + tested.**
4. rung 15 (owed) — operator surgery (C2): install crystallized, gate-passed
   value directions as real operator modes. Mutates the operator; needs its own
   falsification block. **Not built.** This is where the one model becomes literal.
5. H1 human-audit stratum (owed) — the external boundary that makes the rest mean
   anything. Still the top safety dependency.
