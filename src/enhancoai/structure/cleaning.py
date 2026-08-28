"""Structure cleaning: alternate locations, missing-atom reporting, renumbering."""

from __future__ import annotations

import pandas as pd

# Reference heavy-atom counts for completeness reporting (protein backbone + CB is
# always expected; full side-chain complement varies, so we check backbone only,
# which is what most downstream geometry/contact analyses actually require).
BACKBONE_ATOMS = {"N", "CA", "C", "O"}
DNA_BACKBONE_ATOMS = {"P", "O5'", "C5'", "C4'", "C3'", "O3'"}


def resolve_alternate_locations(structure) -> "type(structure)":
    """Keep only the highest-occupancy altloc for each atom position.

    Returns a new StructureData; the input is not mutated.
    """
    from enhancoai.structure.parser import StructureData

    atoms = structure.atoms
    if (atoms["altloc"] == "").all():
        return structure

    def _pick(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) == 1:
            return group
        best_altloc = group.sort_values("occupancy", ascending=False)["altloc"].iloc[0]
        keep = group[(group["altloc"] == best_altloc) | (group["altloc"] == "")]
        return keep.iloc[[0]]

    grouped = atoms.groupby(["model", "chain_id", "res_seq", "icode", "atom_name"], sort=False)
    cleaned = grouped.apply(_pick, include_groups=False).reset_index(drop=True)
    cleaned = cleaned.sort_values("atom_index").reset_index(drop=True)
    return StructureData(atoms=cleaned, source_path=structure.source_path)


def report_missing_backbone_atoms(structure) -> list[str]:
    """Report protein/DNA residues missing expected backbone atoms.

    This is a lightweight completeness check, not a full valence/geometry
    validation, and is intended to flag residues that may bias downstream
    contact or geometry calculations.
    """
    from enhancoai.structure.chain_detection import ChainType, classify_chains

    report: list[str] = []
    classification = classify_chains(structure)
    for chain_id, chain_type in classification.items():
        if chain_type not in (ChainType.PROTEIN, ChainType.DNA):
            continue
        expected = BACKBONE_ATOMS if chain_type == ChainType.PROTEIN else DNA_BACKBONE_ATOMS
        chain_atoms = structure.chain(chain_id)
        for (res_seq, icode), group in chain_atoms.groupby(["res_seq", "icode"]):
            present = set(group["atom_name"].str.strip())
            missing = expected - present
            if missing:
                res_name = group["res_name"].iloc[0]
                report.append(
                    f"Chain {chain_id} {res_name}{res_seq}{icode}: missing {sorted(missing)}"
                )
    return report


def renumber_residues(structure, chain_id: str, start: int = 1) -> "type(structure)":
    """Return a copy of structure with sequential residue numbering for one chain."""
    from enhancoai.structure.parser import StructureData

    atoms = structure.atoms.copy()
    mask = atoms["chain_id"] == chain_id
    residue_keys = (
        atoms.loc[mask, ["res_seq", "icode"]]
        .drop_duplicates()
        .sort_values(["res_seq", "icode"])
        .reset_index(drop=True)
    )
    mapping = {
        (row.res_seq, row.icode): start + i for i, row in enumerate(residue_keys.itertuples())
    }
    new_res_seq = atoms.loc[mask].apply(lambda r: mapping[(r.res_seq, r.icode)], axis=1)
    atoms.loc[mask, "res_seq"] = new_res_seq
    atoms.loc[mask, "icode"] = ""
    return StructureData(atoms=atoms, source_path=structure.source_path)
