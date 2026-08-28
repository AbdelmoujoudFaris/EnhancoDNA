"""TF orientation analysis (section 15): principal axes and inter-molecular angles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from enhancoai.md.loader import TrajectoryHandle


def principal_axes(coords: np.ndarray) -> np.ndarray:
    """Principal axes of a point cloud via PCA of the (mean-centred) covariance matrix.

    Returns a (3, 3) matrix whose columns are the principal axes, ordered
    by decreasing variance (axis 0 = longest/first principal axis).
    """
    centred = coords - coords.mean(axis=0)
    covariance = centred.T @ centred / max(len(coords) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_angle = np.clip(np.abs(np.dot(v1, v2)), -1.0, 1.0)  # axes are directionless
    return float(np.degrees(np.arccos(cos_angle)))


@dataclass
class OrientationFrame:
    frame: int
    time_ps: float
    angle_deg: float


def orientation_trajectory(
    handle: TrajectoryHandle,
    selection_a: str,
    selection_b: str,
    axis_index: int = 0,
) -> pd.DataFrame:
    """Angle (degrees) between the first principal axes of two selections over the trajectory.

    Typical usage: ``selection_a`` = a TF chain's CA atoms, ``selection_b``
    = the DNA helical axis proxy (e.g. all DNA C1' atoms), to track
    theta(TF, DNA) frame by frame; or two protein chains for theta(TF-A, TF-B).
    """
    universe = handle.universe
    atoms_a = universe.select_atoms(selection_a)
    atoms_b = universe.select_atoms(selection_b)
    if len(atoms_a) < 2 or len(atoms_b) < 2:
        raise ValueError("Orientation analysis requires at least 2 atoms per selection.")

    rows = []
    for ts in universe.trajectory[:: handle.stride]:
        axes_a = principal_axes(atoms_a.positions)
        axes_b = principal_axes(atoms_b.positions)
        angle = _angle_between(axes_a[:, axis_index], axes_b[:, axis_index])
        rows.append({"frame": ts.frame, "time_ps": float(ts.time), "angle_deg": angle})
    return pd.DataFrame(rows)


def static_orientation(structure, chain_a: str, chain_b: str, axis_index: int = 0) -> float:
    """Single-structure (no trajectory) orientation angle between two chains' principal axes."""
    coords_a = structure.chain(chain_a)[["x", "y", "z"]].to_numpy(dtype=float)
    coords_b = structure.chain(chain_b)[["x", "y", "z"]].to_numpy(dtype=float)
    axes_a = principal_axes(coords_a)
    axes_b = principal_axes(coords_b)
    return _angle_between(axes_a[:, axis_index], axes_b[:, axis_index])
