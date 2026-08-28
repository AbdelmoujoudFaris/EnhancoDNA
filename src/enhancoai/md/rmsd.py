"""RMSD calculation over a trajectory (protein or DNA selections)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from enhancoai.md.loader import TrajectoryHandle


def compute_rmsd(
    handle: TrajectoryHandle,
    selection: str = "protein and name CA",
    reference_frame: int = 0,
) -> pd.DataFrame:
    """RMSD of ``selection`` to its coordinates at ``reference_frame``.

    Superposition (translation + rotation fit) is performed before the RMSD
    calculation via the Kabsch algorithm, matching the standard definition
    of structural RMSD.
    """
    universe = handle.universe
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"Selection matched no atoms: '{selection}'")

    universe.trajectory[reference_frame]
    reference_coords = atoms.positions.copy()
    reference_coords -= reference_coords.mean(axis=0)

    rows = []
    for ts in universe.trajectory[:: handle.stride]:
        coords = atoms.positions.copy()
        coords -= coords.mean(axis=0)
        rotation = _kabsch_rotation(coords, reference_coords)
        fitted = coords @ rotation
        rmsd = float(np.sqrt(np.mean(np.sum((fitted - reference_coords) ** 2, axis=1))))
        rows.append({"frame": ts.frame, "time_ps": float(ts.time), "rmsd": rmsd})
    return pd.DataFrame(rows)


def _kabsch_rotation(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    covariance = mobile.T @ target
    u, _, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1, 1, d])
    return u @ correction @ vt
