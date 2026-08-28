#!/usr/bin/env python
"""Load, clean and classify a structure; print a summary and optionally save cleaned atoms.

Example:
    python scripts/prepare_structure.py --input data/examples/demo_tf_dna_two_factor.pdb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="PDB or mmCIF file.")
    parser.add_argument("--resolve-altloc", action="store_true", help="Keep only the highest-occupancy altloc.")
    parser.add_argument("--output-csv", default=None, help="Optional path to save the cleaned atom table as CSV.")
    args = parser.parse_args()

    from enhancoai.structure.parser import load_structure
    from enhancoai.structure.chain_detection import summarize_chains
    from enhancoai.structure.cleaning import resolve_alternate_locations, report_missing_backbone_atoms

    structure = load_structure(args.input)
    if args.resolve_altloc:
        structure = resolve_alternate_locations(structure)

    print(f"Loaded {args.input}: {structure.n_atoms()} atoms, {structure.n_models()} model(s).")
    print(summarize_chains(structure).to_string(index=False))

    missing = report_missing_backbone_atoms(structure)
    print(f"\n{len(missing)} residue(s) with missing backbone atoms.")
    for line in missing[:20]:
        print(f"  {line}")

    if args.output_csv:
        structure.atoms.to_csv(args.output_csv, index=False)
        print(f"\nSaved cleaned atom table to {args.output_csv}")


if __name__ == "__main__":
    main()
