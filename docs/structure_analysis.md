# Structure Analysis

## Loading

```python
from enhancoai.structure.parser import load_structure

structure = load_structure("complex.pdb")  # or .cif/.mmcif
```

Returns a `StructureData`: a flat, backend-agnostic atom table (a pandas
DataFrame under `.atoms`) plus convenience accessors (`.chain(chain_id)`,
`.chain_ids`, `.residues()`, `.coords`). Both PDB and mmCIF are parsed via
Biopython; alternate locations, occupancy and B-factors are preserved.

## Chain classification

```python
from enhancoai.structure.chain_detection import classify_chains, summarize_chains

classification = classify_chains(structure)   # {"A": ChainType.PROTEIN, "C": ChainType.DNA, ...}
summarize_chains(structure)                    # DataFrame: chain_id, type, n_residues, n_atoms
```

Classification is by residue-name composition (>=50% standard amino acids
-> Protein, >=50% DA/DT/DG/DC -> DNA, etc.), not by chain ID, so it works
regardless of how a given file labels its chains.

## Cleaning

```python
from enhancoai.structure.cleaning import resolve_alternate_locations, report_missing_backbone_atoms

structure = resolve_alternate_locations(structure)   # keeps highest-occupancy altloc
missing = report_missing_backbone_atoms(structure)   # list[str], backbone-completeness check only
```

`report_missing_backbone_atoms` checks protein backbone (N, CA, C, O) and
DNA backbone (P, O5', C5', C4', C3', O3') presence -- it is a completeness
flag for downstream geometry/contact calculations, not a full
valence/geometry validator.

## Selection helpers

`enhancoai.structure.selection` provides `ca_atoms`, `dna_phosphate_atoms`,
`dna_base_atoms`, `heavy_atoms`, `chain_center_of_mass`, etc. -- all take
and return plain atom DataFrames so they compose with pandas directly.
