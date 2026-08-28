"""Protein-protein (TF-TF) interaction analysis (section 11 of the spec)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from enhancoai.interactions.contact_maps import find_atom_contacts, contacts_to_frame, residue_contact_map
from enhancoai.interactions.hydrogen_bonds import find_hydrogen_bonds, find_salt_bridges
from enhancoai.structure.selection import heavy_atoms


@dataclass
class ProteinProteinInteractionResult:
    chain_a: str
    chain_b: str
    residue_contact_map: pd.DataFrame
    hydrogen_bonds: pd.DataFrame
    salt_bridges: pd.DataFrame
    min_distance: float
    interface_area_a2: float | None
    buried_surface_area_a2: float | None

    def summary(self) -> dict:
        return {
            "chain_a": self.chain_a,
            "chain_b": self.chain_b,
            "n_interface_residues_a": self.residue_contact_map["res_seq_a"].nunique()
            if not self.residue_contact_map.empty
            else 0,
            "n_interface_residues_b": self.residue_contact_map["res_seq_b"].nunique()
            if not self.residue_contact_map.empty
            else 0,
            "min_distance": self.min_distance,
            "n_hydrogen_bonds": len(self.hydrogen_bonds),
            "n_salt_bridges": len(self.salt_bridges),
            "buried_surface_area_a2": self.buried_surface_area_a2,
        }


def _sasa_total(structure_model, chain_ids: list[str] | None = None) -> float | None:
    """Total solvent-accessible surface area (Shrake-Rupley) via Biopython.

    Returns None if the SASA implementation is unavailable in the installed
    Biopython version, in which case buried-surface-area estimates are
    reported as unavailable rather than fabricated.
    """
    try:
        from Bio.PDB.SASA import ShrakeRupley
    except ImportError:
        return None

    sr = ShrakeRupley()
    sr.compute(structure_model, level="A")
    total = 0.0
    for chain in structure_model:
        if chain_ids is not None and chain.id not in chain_ids:
            continue
        for residue in chain:
            for atom in residue:
                total += atom.sasa
    return total


def _buried_surface_area(structure, chain_a: str, chain_b: str) -> tuple[float | None, float | None]:
    """Estimate interface (buried) surface area as SASA(A) + SASA(B) - SASA(A+B).

    Requires Biopython's ShrakeRupley SASA implementation and re-parses the
    structure via Bio.PDB to obtain a hierarchical model object.
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError:
        return None, None

    if not structure.source_path or not structure.source_path.lower().endswith((".pdb", ".ent")):
        return None, None

    parser = PDBParser(QUIET=True)
    bio_structure = parser.get_structure("complex", structure.source_path)
    model = next(iter(bio_structure))

    sasa_a = _sasa_total(model, [chain_a])
    sasa_b = _sasa_total(model, [chain_b])
    sasa_ab = _sasa_total(model, [chain_a, chain_b])
    if None in (sasa_a, sasa_b, sasa_ab):
        return None, None

    buried = sasa_a + sasa_b - sasa_ab
    interface_area = buried / 2.0
    return interface_area, buried


def analyse_protein_protein_interactions(
    structure,
    chain_a: str,
    chain_b: str,
    heavy_atom_cutoff: float = 5.0,
    hydrogen_bond_cutoff: float = 3.5,
    salt_bridge_cutoff: float = 4.0,
    compute_sasa: bool = True,
) -> ProteinProteinInteractionResult:
    atoms_a = heavy_atoms(structure.chain(chain_a))
    atoms_b = heavy_atoms(structure.chain(chain_b))

    contacts = find_atom_contacts(atoms_a, atoms_b, cutoff=heavy_atom_cutoff)
    res_map = residue_contact_map(contacts)
    hbonds = find_hydrogen_bonds(atoms_a, atoms_b, distance_cutoff=hydrogen_bond_cutoff)
    salt_bridges = find_salt_bridges(atoms_a, atoms_b, distance_cutoff=salt_bridge_cutoff)

    frame = contacts_to_frame(contacts)
    min_distance = float(frame["distance"].min()) if not frame.empty else float("inf")

    interface_area, buried_area = (None, None)
    if compute_sasa:
        interface_area, buried_area = _buried_surface_area(structure, chain_a, chain_b)

    return ProteinProteinInteractionResult(
        chain_a=chain_a,
        chain_b=chain_b,
        residue_contact_map=res_map,
        hydrogen_bonds=hbonds,
        salt_bridges=salt_bridges,
        min_distance=min_distance,
        interface_area_a2=interface_area,
        buried_surface_area_a2=buried_area,
    )


def contact_frequency_over_trajectory(per_frame_results: list[ProteinProteinInteractionResult]) -> pd.DataFrame:
    """Interaction persistence of each interface residue pair across frames."""
    n = len(per_frame_results)
    if n == 0:
        return pd.DataFrame(columns=["res_seq_a", "res_seq_b", "frequency"])
    counts: dict[tuple, int] = {}
    for result in per_frame_results:
        for row in result.residue_contact_map.itertuples():
            key = (row.res_seq_a, row.res_seq_b)
            counts[key] = counts.get(key, 0) + 1
    rows = [{"res_seq_a": k[0], "res_seq_b": k[1], "frequency": v / n} for k, v in counts.items()]
    return pd.DataFrame(rows).sort_values("frequency", ascending=False).reset_index(drop=True)
