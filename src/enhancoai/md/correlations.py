"""Dynamic cross-correlation (DCCM) and mutual information between residues.

DCCM measures *linear* correlation of residue displacement vectors from
the mean structure. Mutual information additionally captures nonlinear
coupling but requires substantially more frames to estimate reliably; see
:mod:`enhancoai.allostery.mutual_information` for the graph-construction
use of this output, and the limitations documentation for sample-size
caveats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from enhancoai.md.loader import TrajectoryHandle
from enhancoai.md.rmsd import _kabsch_rotation


def _superposed_displacements(handle: TrajectoryHandle, selection: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    universe = handle.universe
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"Selection matched no atoms: '{selection}'")

    all_coords = []
    for ts in universe.trajectory[:: handle.stride]:
        coords = atoms.positions.copy()
        coords -= coords.mean(axis=0)
        all_coords.append(coords)
    all_coords = np.array(all_coords)

    reference = all_coords[0]
    fitted = np.empty_like(all_coords)
    for i, coords in enumerate(all_coords):
        rotation = _kabsch_rotation(coords, reference)
        fitted[i] = coords @ rotation

    mean_structure = fitted.mean(axis=0)
    displacements = fitted - mean_structure  # (n_frames, n_atoms, 3)
    return displacements, atoms.resids, atoms.resnames


def dynamic_cross_correlation(handle: TrajectoryHandle, selection: str = "protein and name CA") -> pd.DataFrame:
    """Normalised dynamic cross-correlation matrix, indexed by residue id.

    C_ij = <dr_i . dr_j> / sqrt(<dr_i^2> <dr_j^2>), in [-1, 1].
    """
    displacements, resids, _ = _superposed_displacements(handle, selection)
    n_frames, n_atoms, _ = displacements.shape

    dot = np.einsum("fik,fjk->ij", displacements, displacements) / n_frames
    variances = np.diag(dot)
    denom = np.sqrt(np.outer(variances, variances)) + 1e-12
    correlation = dot / denom

    return pd.DataFrame(correlation, index=resids, columns=resids)


def mutual_information_matrix(handle: TrajectoryHandle, selection: str = "protein and name CA", n_bins: int = 10) -> pd.DataFrame:
    """Pairwise mutual information (nats) between residue displacement magnitudes.

    Uses a coarse histogram-based estimator on the scalar displacement
    magnitude per residue per frame. This discards directional information
    (unlike DCCM) but detects nonlinear coupling; it requires many frames
    per bin to be reliable and is documented as exploratory.
    """
    displacements, resids, _ = _superposed_displacements(handle, selection)
    magnitudes = np.linalg.norm(displacements, axis=2)  # (n_frames, n_atoms)
    n_atoms = magnitudes.shape[1]

    mi = np.zeros((n_atoms, n_atoms))
    digitised = np.array(
        [
            np.digitize(magnitudes[:, i], np.histogram(magnitudes[:, i], bins=n_bins)[1][1:-1])
            for i in range(n_atoms)
        ]
    )  # (n_atoms, n_frames)

    for i in range(n_atoms):
        for j in range(i, n_atoms):
            mi_val = _discrete_mutual_information(digitised[i], digitised[j], n_bins)
            mi[i, j] = mi_val
            mi[j, i] = mi_val

    return pd.DataFrame(mi, index=resids, columns=resids)


def _discrete_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    joint = np.zeros((n_bins, n_bins))
    for xi, yi in zip(x, y):
        joint[min(xi, n_bins - 1), min(yi, n_bins - 1)] += 1
    joint /= joint.sum() + 1e-12
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px * py + 1e-12)
        terms = joint * np.log(np.where(ratio > 0, ratio, 1.0))
    return float(np.nansum(terms))
