"""Per-residue RMSF (root-mean-square fluctuation) over a trajectory."""

from __future__ import annotations

import numpy as np
import pandas as pd

from enhancoai.md.loader import TrajectoryHandle
from enhancoai.md.rmsd import _kabsch_rotation


def compute_rmsf(handle: TrajectoryHandle, selection: str = "protein and name CA") -> pd.DataFrame:
    """RMSF per atom in ``selection``, after superposition to the mean structure.

    Two-pass algorithm: (1) accumulate superposed coordinates to build a
    mean reference structure, (2) compute per-atom fluctuation about that
    mean.
    """
    universe = handle.universe
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"Selection matched no atoms: '{selection}'")

    all_coords = []
    for ts in universe.trajectory[:: handle.stride]:
        coords = atoms.positions.copy()
        coords -= coords.mean(axis=0)
        all_coords.append(coords)
    all_coords = np.array(all_coords)  # (n_frames, n_atoms, 3)

    reference = all_coords[0]
    fitted = np.empty_like(all_coords)
    for i, coords in enumerate(all_coords):
        rotation = _kabsch_rotation(coords, reference)
        fitted[i] = coords @ rotation

    mean_structure = fitted.mean(axis=0)
    deviations = fitted - mean_structure
    rmsf = np.sqrt(np.mean(np.sum(deviations**2, axis=2), axis=0))

    resids = atoms.resids
    resnames = atoms.resnames
    return pd.DataFrame({"res_seq": resids, "res_name": resnames, "rmsf": rmsf})
