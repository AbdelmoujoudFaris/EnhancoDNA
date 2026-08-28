# Data Directory

- `raw/` -- untouched input structures/trajectories you supply (gitignored).
- `processed/` -- feature tensors and datasets built by `scripts/build_dataset.py`
  or `scripts/generate_demo_dataset.py` (gitignored).
- `examples/` -- small, committed example files, including the synthetic
  demo complexes produced by `scripts/generate_demo_dataset.py`
  (`demo_tf_dna_two_factor.pdb`, `demo_tf_dna_single_factor.pdb`).

## Licensing

Do not commit copyrighted or restricted-access experimental structures or
trajectories to this repository. Reference external structures by their
public accession (e.g. a PDB ID) and let users fetch them themselves, or
use the synthetic demo generator for anything that needs to ship with the
repo.

## DEMO DATA warning

Anything under `data/examples/` produced by
`scripts/generate_demo_dataset.py` is procedurally generated and
explicitly **NOT SCIENTIFICALLY VALIDATED** -- it exists only to exercise
the full pipeline (structure loading through AI training) without
depending on external data.
