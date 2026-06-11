# gue_domain_tester.py — usage guide

Domain-aware GUE testing for any kind of data. Tests whether a
dataset's spacing structure matches the GUE nearest-neighbour
spacing distribution, using preprocessing appropriate to the domain.

## Why domain-specific preprocessing matters

The naive approach — sort values, compute differences, normalize —
only works for data with constant density. Real datasets have
structure that must be removed before testing for GUE:

  SPECTRAL  Eigenvalues cluster near the centre following the Wigner
            semicircle law. Without correcting for this, you test the
            semicircle density, not the GUE spacing structure.

  TEMPORAL  Event rates vary over time. A slow period creates large
            inter-event gaps that look like non-GUE structure but are
            just a rate change. The KDE-based unfolding removes this.

  SPATIAL   Same as temporal but along a spatial axis. Local density
            must be estimated and divided out.

  LINGUISTIC Feature positions in text (stressed syllables, rhyme
            words, sentence boundaries). Raw character positions are
            appropriate after spacing computation.

  FINANCIAL Log-returns often need volatility normalization first —
            heteroskedasticity (varying variance) would otherwise
            obscure any GUE structure.

  GENERIC   Simple mean-normalised NNS. Use when no domain structure
            is known.

## Verified behaviour

```
★ GUE 80×80 matrix          GUE=0.905   closest=GUE
  Random eigenvalues         GUE=0.653   closest=Poisson
★ GUE-spaced events          GUE=0.945   closest=GUE
  Poisson arrivals           GUE=0.673   closest=Poisson
```

Rolling window correctly tracked GUE → Poisson → GUE transition,
score dropping from 0.94 to 0.67 in the Poisson region and
recovering to 0.90 in the second GUE region.

Multi-scale test showed GUE persisting from scale=2 through scale=32
with the GUE signal gradually weakening at coarser scales.

## Basic usage

```python
from gue_domain_tester import (
    DomainGUETester, SPECTRAL, TEMPORAL, SPATIAL_1D,
    LINGUISTIC, FINANCIAL, GENERIC,
)

tester = DomainGUETester(n_bootstrap=300, min_spacings=8)

# Test a matrix
result = tester.test(matrix, domain=SPECTRAL, label="Hamiltonian")

# Test timestamps
result = tester.test(event_times, domain=TEMPORAL, label="Neural spikes")

# Test text
result = tester.test(
    text, domain=LINGUISTIC, label="Rhyme positions",
    preprocess_kwargs={"feature": "newlines"},
)

# Pre-extracted positions (rhyme word positions, etc.)
result = tester.test(
    positions, domain=LINGUISTIC,
    preprocess_kwargs={"feature": "positions"},
)
```

## Shakespeare analysis

For your rhyme spacing analysis:

```python
# Extract positions yourself (word index of each rhyme word)
rhyme_positions = [12, 24, 25, 37, 49, 50, ...]  # from your analysis

result = tester.test(
    rhyme_positions, domain=LINGUISTIC,
    label="Hamlet rhyme spacings",
    preprocess_kwargs={"feature": "positions"},
)

# Or test at multiple linguistic levels
for feature in ["newlines", "words", "sentences"]:
    tester.test(text, domain=LINGUISTIC,
                label=f"Shakespeare ({feature})",
                preprocess_kwargs={"feature": feature})
```

## Rolling window — detecting WHERE GUE emerges

```python
positions, scores = tester.rolling_gue_trajectory(
    event_times,
    domain=TEMPORAL,
    window_size=100,    # events per window
    step=20,            # step between windows
)

# scores[i] = GUE score at positions[i]
# Plot or inspect to see where GUE is strong vs. weak
```

Useful for:
- Which act of a Shakespeare play is most GUE-like
- Which period of financial data shows GUE statistics
- Which brain state (rest vs. task) is more GUE-like

## Multi-scale — detecting WHICH SCALE GUE emerges at

```python
scale_results = tester.test_multiscale(
    positions,
    domain=TEMPORAL,
    scales=[1, 2, 4, 8, 16, 32],
    label="My data",
)

# Shows GUE score at each coarsening level
# Tells you: is GUE structure fine-grained or coarse-grained?
```

## Multi-domain comparison

```python
results = [
    tester.test(eigenvalues, domain=SPECTRAL,   label="Hamiltonian"),
    tester.test(spike_times, domain=TEMPORAL,   label="Neural spikes"),
    tester.test(rhyme_pos,   domain=LINGUISTIC, label="Rhyme spacings"),
    tester.test(log_returns, domain=FINANCIAL,  label="Market returns"),
]
report = tester.compare(results, label="Cross-domain GUE test")
print(report.summary())
# Shows ranked table of GUE scores across all domains
```

## Integration with gue_engine.py

The domain tester imports from `gue_engine.py`. Both files need to be
in the same directory. `gue_engine.py` provides the core math;
`gue_domain_tester.py` handles the preprocessing.

## Important notes on sample size

GUE statistics require spacing distributions. With N data points you
get N-1 spacings. The bootstrap CI becomes reliable only for N ≥ 30
(the tester warns about small samples). For rolling windows, a window
size of 50-100 is a reasonable minimum.

## Extending to new domains

Add a preprocessing function that returns the domain-appropriate
VALUES (positions) in a form where nearest-neighbour spacing is
meaningful. The function should remove domain-specific density
variation so the spacing reflects structural organization rather than
background variation.

```python
def preprocess_mydomain(data, **kwargs) -> np.ndarray:
    # ... domain-specific transformation ...
    return corrected_positions  # identify_ensemble computes spacings

from gue_domain_tester import PREPROCESSORS
PREPROCESSORS["mydomain"] = preprocess_mydomain
```
