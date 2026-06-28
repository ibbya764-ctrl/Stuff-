# BHDC GPU runs — pre-registered, matched-budget

Ready to launch on the workstation (RTX 4060 Ti / Modal A100). The runner picks
CUDA automatically. Every config carries a written prediction + kill condition;
`bhdc_runner.py` **refuses** to run without them, and refuses any A/B whose arms
differ in parameter count by >2% (matched-budget rule).

## Launch

```bash
# headline existence claim: does BHDC grok (a+b) mod 97?
python3 bhdc_runner.py configs/bhdc_grok_p97.json

# the three attribution A/Bs (each writes results/<name>.json)
python3 bhdc_runner.py configs/bhdc_collapse_ab.json    # coherence is the mechanism
python3 bhdc_runner.py configs/bhdc_geometry_ab.json    # does §4 GR coupling earn its place
python3 bhdc_runner.py configs/bhdc_entangle_ab.json    # does §5 entanglement beat a capacity-matched control
```

Each prints per-seed and mean±std best val accuracy, telemetry, gate warnings,
deltas vs the baseline arm, and the pre-registered prediction/kill text beside
the numbers. Results are also saved as JSON under `results/`.

## The experiments

| config | arms | matched? | the question |
|---|---|---|---|
| `bhdc_grok_p97` | coherent ×3 seeds | n/a (existence vs chance) | Does it grok at all (val ≫ 2/p)? |
| `bhdc_collapse_ab` | single / incoherent / coherent / coherent-complex | **exact** (identical params) | Is **coherence** the mechanism, not multi-scenario averaging? |
| `bhdc_geometry_ab` | core / +geometry | +0.004% | Does the §4 GR coupling speed/raise grokking? |
| `bhdc_entangle_ab` | core+capacity / +entangle(eigen) / +entangle(generic) | +0.09% | Does §5 entanglement beat a **capacity-matched** control, and does the operator-eigenbasis beat a generic ladder? |

`bhdc_collapse_ab` is the primary result — it saturates the CPU pilot's onset
signal (coherent 0.064 vs single/incoherent 0.000 at 1500 steps) to a full,
multi-seed grok. The entanglement A/B's control is **not** bare core but a
generic residual MLP (`capacity_pad=18`) sized to the entanglement arm's budget,
so a win attributes to the entangling *structure*, not the extra parameters.

## Budget / sizing notes

- Defaults: d_model 128, 2 layers, 4 branches, 30k steps, full-batch, AdamW
  lr 1e-3, wd 1.0 (grokking wants high weight decay), 50% split, 3 seeds.
- p=97 → 9 409 pairs, 4 704 train. Full-batch fits easily; a full A/B (arms ×
  3 seeds × 30k steps) is minutes-to-low-tens-of-minutes on the 4060 Ti.
- To scale toward the §9 ~100M run: raise `d_model` (≈720) and `n_layers` (≈14)
  in the `model` block; the matched-budget guard still applies per A/B.
- Override device with `--device cuda|cpu`; override the output path with `--out`.

## Reading the verdict

The runner surfaces the numbers the kill condition references; the verdict is
made by reading them against the written prediction (no post-hoc softening). A
clean negative is a result — record it in `CP2_AI_Architecture_v3.md` with a
`[TESTED-NEGATIVE]` label.
