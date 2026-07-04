# H1 human-audit stratum

The external boundary condition (v18 §7 / addendum §3). `h1_seed_judgments.jsonl`
holds human pairwise differential judgments (easier for raters than absolute
scores) and a few absolute anchors. Each renewal cycle, a small batch of these
is ingested into the mode bank via `bhdc_icl.audit_stratum.AuditStratum`, which:

- writes an `AuditRecord` onto each judged mode's `audit_history`;
- records ledger entries on the `audit` channel (real audit magnitude, latched);
- on disagreement, ADDS scrutiny (re-derivation pressure + deny bias) — one-way.

The descriptions here are prompt/behaviour families; a live run maps them onto
concrete mode lineages by matching generated candidates, then ingests by
`lineage_id`. This file is the seed; the standing item is 20–50 fresh pairwise
judgments per renewal cycle. The audit stays OUTSIDE the model at every stage.
