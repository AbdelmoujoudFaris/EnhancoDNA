"""Protein-DNA interaction analysis (section 9 of the EnhancoAI spec).

Produces protein-residue <-> DNA-nucleotide contact tables covering
heavy-atom contacts, hydrogen bonds, salt bridges and a simplified van der
Waals classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from enhancoai.interactions.contact_maps import (
    find_atom_contacts,
    contacts_to_frame,
    residue_contact_map,
)
from enhancoai.interactions.hydrogen_bonds import find_hydrogen_bonds, find_salt_bridges
from enhancoai.structure.selection import heavy_atoms

# Approximate van der Waals radii (Angstrom) for common elements.
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.20}


@dataclass
class ProteinDNAInteractionResult:
    protein_chain: str
    dna_chain: str
    heavy_atom_contacts: pd.DataFrame
    residue_contact_map: pd.DataFrame
    hydrogen_bonds: pd.DataFrame
    salt_bridges: pd.DataFrame
    van_der_waals_contacts: pd.DataFrame

    def summary(self) -> dict:
        return {
            "protein_chain": self.protein_chain,
            "dna_chain": self.dna_chain,
            "n_interface_protein_residues": self.residue_contact_map["res_seq_a"].nunique()
            if not self.residue_contact_map.empty
            else 0,
            "n_contacted_dna_bases": self.residue_contact_map["res_seq_b"].nunique()
            if not self.residue_contact_map.empty
            else 0,
            "n_hydrogen_bonds": len(self.hydrogen_bonds),
            "n_salt_bridges": len(self.salt_bridges),
        }


def _van_der_waals_contacts(atoms_a: pd.DataFrame, atoms_b: pd.DataFrame, tolerance: float = 0.5) -> pd.DataFrame:
    """Heavy atoms within (sum of vdW radii + tolerance) of each other."""
    max_cutoff = 2 * max(VDW_RADII.values()) + tolerance
    contacts = find_atom_contacts(atoms_a, atoms_b, cutoff=max_cutoff)
    frame = contacts_to_frame(contacts)
    if frame.empty:
        return frame

    elem_a = atoms_a.set_index(atoms_a["atom_name"])  # not used directly; kept for clarity
    radii_a = atoms_a.assign(radius=atoms_a["element"].str.upper().map(VDW_RADII).fillna(1.7))
    radii_b = atoms_b.assign(radius=atoms_b["element"].str.upper().map(VDW_RADII).fillna(1.7))

    def _radius(chain, res_seq, atom_name, table):
        match = table[
            (table["chain_id"] == chain) & (table["res_seq"] == res_seq) & (table["atom_name"] == atom_name)
        ]
        return float(match["radius"].iloc[0]) if len(match) else 1.7

    thresholds = [
        _radius(r.chain_a, r.res_seq_a, r.atom_a, radii_a) + _radius(r.chain_b, r.res_seq_b, r.atom_b, radii_b) + tolerance
        for r in frame.itertuples()
    ]
    frame = frame.assign(vdw_threshold=thresholds)
    return frame[frame["distance"] <= frame["vdw_threshold"]].drop(columns=["vdw_threshold"])


def analyse_protein_dna_interactions(
    structure,
    protein_chain: str,
    dna_chain: str,
    heavy_atom_cutoff: float = 5.0,
    hydrogen_bond_cutoff: float = 3.5,
    salt_bridge_cutoff: float = 4.0,
) -> ProteinDNAInteractionResult:
    """Full protein-DNA interaction characterisation for one chain pair."""
    protein_atoms = heavy_atoms(structure.chain(protein_chain))
    dna_atoms = heavy_atoms(structure.chain(dna_chain))

    heavy_contacts = find_atom_contacts(protein_atoms, dna_atoms, cutoff=heavy_atom_cutoff)
    heavy_frame = contacts_to_frame(heavy_contacts)
    res_map = residue_contact_map(heavy_contacts)

    hbonds = find_hydrogen_bonds(protein_atoms, dna_atoms, distance_cutoff=hydrogen_bond_cutoff)
    salt_bridges = find_salt_bridges(protein_atoms, dna_atoms, distance_cutoff=salt_bridge_cutoff)
    vdw = _van_der_waals_contacts(protein_atoms, dna_atoms)

    return ProteinDNAInteractionResult(
        protein_chain=protein_chain,
        dna_chain=dna_chain,
        heavy_atom_contacts=heavy_frame,
        residue_contact_map=res_map,
        hydrogen_bonds=hbonds,
        salt_bridges=salt_bridges,
        van_der_waals_contacts=vdw,
    )


def format_contact_descriptions(result: ProteinDNAInteractionResult) -> list[str]:
    """Human-readable 'TF-A ARG42 -> DNA G18' style descriptions."""
    lines = []
    for row in result.residue_contact_map.itertuples():
        lines.append(
            f"{result.protein_chain} {row.res_name_a}{row.res_seq_a} -> "
            f"{result.dna_chain} {row.res_name_b}{row.res_seq_b} "
            f"(min dist {row.min_distance:.2f} A, {row.n_atom_contacts} atom contacts)"
        )
    return lines
