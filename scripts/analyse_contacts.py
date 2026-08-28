#!/usr/bin/env python
"""Full protein-DNA and protein-protein contact analysis for one structure.

Example:
    python scripts/analyse_contacts.py --input data/examples/demo_tf_dna_two_factor.pdb \\
        --protein A --protein B --dna C --dna D --output-dir experiments/contacts_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--protein", action="append", required=True, dest="protein_chains")
    parser.add_argument("--dna", action="append", required=True, dest="dna_chains")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--output-dir", default="experiments/contacts")
    args = parser.parse_args()

    from enhancoai.structure.parser import load_structure
    from enhancoai.interactions.protein_dna import analyse_protein_dna_interactions
    from enhancoai.interactions.protein_protein import analyse_protein_protein_interactions
    from enhancoai.visualization.contacts import render_contact_map

    structure = load_structure(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for protein_chain in args.protein_chains:
        for dna_chain in args.dna_chains:
            result = analyse_protein_dna_interactions(structure, protein_chain, dna_chain, heavy_atom_cutoff=args.cutoff)
            print(f"{protein_chain} <-> {dna_chain}: {result.summary()}")
            result.residue_contact_map.to_csv(output_dir / f"contacts_{protein_chain}_{dna_chain}.csv", index=False)
            fig = render_contact_map(result.residue_contact_map, title=f"{protein_chain} <-> {dna_chain}")
            fig.savefig(output_dir / f"contacts_{protein_chain}_{dna_chain}.png", dpi=120)

    for i, chain_a in enumerate(args.protein_chains):
        for chain_b in args.protein_chains[i + 1 :]:
            result = analyse_protein_protein_interactions(structure, chain_a, chain_b, heavy_atom_cutoff=args.cutoff)
            print(f"{chain_a} <-> {chain_b} (protein-protein): {result.summary()}")

    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
