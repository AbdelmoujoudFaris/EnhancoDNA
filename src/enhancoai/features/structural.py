"""Static structural features of a protein chain."""

from __future__ import annotations

import numpy as np


def radius_of_gyration(coords: np.ndarray) -> float:
    center = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))


def chain_radius_of_gyration(structure, chain_id: str) -> float:
    coords = structure.chain(chain_id)[["x", "y", "z"]].to_numpy(dtype=float)
    return radius_of_gyration(coords)


def secondary_structure_fractions(structure_path: str, chain_id: str) -> dict[str, float] | None:
    """Helix/sheet/coil fraction via DSSP, if the ``mkdssp`` binary is available.

    Returns None (not fabricated fractions) when DSSP is not installed, per
    the "no fake science" requirement.
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
    except ImportError:
        return None

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", structure_path)
        model = next(iter(structure))
        dssp = DSSP(model, structure_path)
    except Exception:
        return None

    codes = [dssp[key][2] for key in dssp.keys() if key[0] == chain_id]
    if not codes:
        return None
    n = len(codes)
    helix = sum(1 for c in codes if c in "HGI") / n
    sheet = sum(1 for c in codes if c in "EB") / n
    coil = 1.0 - helix - sheet
    return {"helix": helix, "sheet": sheet, "coil": coil}
