"""Atom/residue selection helpers built on top of StructureData."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_chains(structure, chain_ids: list[str]) -> pd.DataFrame:
    return structure.atoms[structure.atoms["chain_id"].isin(chain_ids)]


def select_model(structure, model: int = 1) -> pd.DataFrame:
    return structure.atoms[structure.atoms["model"] == model]


def ca_atoms(structure, chain_id: str) -> pd.DataFrame:
    """Alpha-carbon atoms of a protein chain (one per residue)."""
    chain = structure.chain(chain_id)
    return chain[chain["atom_name"].str.strip() == "CA"]


def dna_phosphate_atoms(structure, chain_id: str) -> pd.DataFrame:
    chain = structure.chain(chain_id)
    return chain[chain["atom_name"].str.strip() == "P"]


def dna_base_atoms(structure, chain_id: str) -> pd.DataFrame:
    """Return DNA base (non-backbone, non-sugar) atoms of a chain."""
    backbone = {
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
    }
    chain = structure.chain(chain_id)
    return chain[~chain["atom_name"].str.strip().isin(backbone)]


def heavy_atoms(atoms: pd.DataFrame) -> pd.DataFrame:
    """Filter hydrogens out of an atom table (e.g. from ``structure.chain(chain_id)``)."""
    return atoms[atoms["element"].str.upper() != "H"]


def residue_center_of_mass(atom_group: pd.DataFrame) -> np.ndarray:
    return atom_group[["x", "y", "z"]].to_numpy(dtype=float).mean(axis=0)


def chain_center_of_mass(structure, chain_id: str) -> np.ndarray:
    return residue_center_of_mass(structure.chain(chain_id))
