# Reproducibility and Statistical Limitations

## What gets recorded

```python
from enhancoai.utils.reproducibility import ReproducibilityRecord, set_global_seed

set_global_seed(42)
record = ReproducibilityRecord()
record.add_input("structure", "complex.pdb")   # sha256 hash
record.parameters = {"protein_dna_cutoff": 5.0}
record.export("reproducibility.json")
```

Captures: input file content hashes (sha256), analysis parameters,
installed software versions (`numpy`, `torch`, `MDAnalysis`, `Bio`,
`PySide6`, ...), CUDA version if applicable, random seed, and a UTC
timestamp. `scripts/generate_report.py` writes one automatically per
report.

## Statistical limitations (read before trusting a number)

- **MD frames are not independent samples.** Consecutive frames are
  temporally correlated; do not treat frame count as replicate count when
  computing confidence intervals. Use block averaging or independent
  replicate trajectories instead.
- **PMFs assume equilibrium, unbiased sampling.** `compute_pmf_1d`/`_2d`
  warn when sample counts are low, and mark under-sampled bins `NaN`
  rather than extrapolating -- but a warning-free PMF is not automatically
  a converged one if the underlying simulation itself did not sample the
  full coordinate range.
- **Correlation is not causation.** DCCM and mutual-information edges in
  the allosteric network are statistical associations; corroborate with
  independent evidence types (contacts, free energy, orientation) before
  concluding a causal pathway.
- **AI predictions depend on training data.** A model trained on a small
  or narrow dataset (like the bundled demo) will not generalise; treat its
  predictions as illustrative only until trained/validated on real,
  sufficiently diverse, correctly-split data (see
  `scripts/build_dataset.py`'s clustered-split logic).
- **Reaction-coordinate choice matters.** A PMF's shape and derived
  binding free energy depend on the coordinate you chose; state that
  choice whenever reporting a result.

## Export Reproducibility Package

`ReproducibilityRecord.export(path)` writes the full record as JSON --
attach it alongside any figure or number you report from EnhancoAI.
