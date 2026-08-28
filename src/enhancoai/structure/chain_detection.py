"""Automatic classification of chains into Protein / DNA / RNA / Ligand / Other.

Classification is residue-composition based so it works for any TF/DNA
complex without hard-coding specific chain IDs.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}

DNA_RESIDUES = {"DA", "DT", "DG", "DC", "DI", "DU"}
RNA_RESIDUES = {"A", "U", "G", "C", "I"}
WATER_RESIDUES = {"HOH", "WAT", "H2O"}


class ChainType(str, Enum):
    PROTEIN = "Protein"
    DNA = "DNA"
    RNA = "RNA"
    LIGAND = "Ligand"
    WATER = "Water"
    OTHER = "Other"


def classify_chain(residue_names: list[str]) -> ChainType:
    """Classify a single chain from the residue names it contains."""
    names = [r.strip().upper() for r in residue_names if r.strip().upper() not in WATER_RESIDUES]
    if not names:
        return ChainType.WATER

    n = len(names)
    n_protein = sum(1 for r in names if r in STANDARD_AMINO_ACIDS)
    n_dna = sum(1 for r in names if r in DNA_RESIDUES)
    n_rna = sum(1 for r in names if r in RNA_RESIDUES and r not in DNA_RESIDUES)

    if n_protein / n >= 0.5:
        return ChainType.PROTEIN
    if n_dna / n >= 0.5:
        return ChainType.DNA
    if n_rna / n >= 0.5:
        return ChainType.RNA
    if n <= 3:
        return ChainType.LIGAND
    return ChainType.OTHER


def classify_chains(structure) -> dict[str, ChainType]:
    """Classify every chain in a :class:`~enhancoai.structure.parser.StructureData`.

    Returns a mapping ``chain_id -> ChainType``.
    """
    result: dict[str, ChainType] = {}
    for chain_id in structure.chain_ids:
        residues = structure.chain(chain_id)["res_name"].tolist()
        result[chain_id] = classify_chain(residues)
    return result


def summarize_chains(structure) -> pd.DataFrame:
    """Return a per-chain summary table (chain_id, type, n_residues, n_atoms)."""
    classification = classify_chains(structure)
    rows = []
    for chain_id, chain_type in classification.items():
        chain_atoms = structure.chain(chain_id)
        n_residues = chain_atoms.drop_duplicates(subset=["res_seq", "icode", "model"]).shape[0]
        rows.append(
            {
                "chain_id": chain_id,
                "type": chain_type.value,
                "n_residues": n_residues,
                "n_atoms": len(chain_atoms),
            }
        )
    return pd.DataFrame(rows)
