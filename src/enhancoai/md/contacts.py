"""Trajectory-level contact persistence and interface-area tracking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from enhancoai.md.loader import TrajectoryHandle


def contact_persistence(
    handle: TrajectoryHandle,
    selection_a: str,
    selection_b: str,
    cutoff: float = 5.0,
) -> pd.DataFrame:
    """Fraction of frames each (resid_a, resid_b) pair is within ``cutoff`` A.

    Operates at atom resolution internally (any-atom-in-contact) but
    reports at residue resolution, consistent with
    :func:`enhancoai.interactions.contact_maps.contact_frequency`.
    """
    universe = handle.universe
    atoms_a = universe.select_atoms(selection_a)
    atoms_b = universe.select_atoms(selection_b)
    if len(atoms_a) == 0 or len(atoms_b) == 0:
        raise ValueError("Empty selection in contact_persistence().")

    counts: dict[tuple, int] = {}
    n_frames = 0
    for ts in universe.trajectory[:: handle.stride]:
        n_frames += 1
        tree_b = cKDTree(atoms_b.positions)
        pairs = cKDTree(atoms_a.positions).query_ball_tree(tree_b, r=cutoff)
        seen = set()
        for i, neighbours in enumerate(pairs):
            if not neighbours:
                continue
            resid_a = int(atoms_a.resids[i])
            for j in neighbours:
                resid_b = int(atoms_b.resids[j])
                seen.add((resid_a, resid_b))
        for key in seen:
            counts[key] = counts.get(key, 0) + 1

    if n_frames == 0:
        return pd.DataFrame(columns=["resid_a", "resid_b", "frequency"])

    rows = [{"resid_a": k[0], "resid_b": k[1], "frequency": v / n_frames} for k, v in counts.items()]
    return pd.DataFrame(rows).sort_values("frequency", ascending=False).reset_index(drop=True)


def center_of_mass_distance(handle: TrajectoryHandle, selection_a: str, selection_b: str) -> pd.DataFrame:
    universe = handle.universe
    atoms_a = universe.select_atoms(selection_a)
    atoms_b = universe.select_atoms(selection_b)
    rows = []
    for ts in universe.trajectory[:: handle.stride]:
        com_a = atoms_a.center_of_mass()
        com_b = atoms_b.center_of_mass()
        rows.append({"frame": ts.frame, "time_ps": float(ts.time), "com_distance": float(np.linalg.norm(com_a - com_b))})
    return pd.DataFrame(rows)


def hydrogen_bond_count_trajectory(handle: TrajectoryHandle, selection_a: str, selection_b: str, distance_cutoff: float = 3.5) -> pd.DataFrame:
    """Frame-by-frame count of polar heavy-atom pairs (N/O/S) within ``distance_cutoff``.

    Heavy-atom-distance proxy for hydrogen bonds; see
    :mod:`enhancoai.interactions.hydrogen_bonds` for the same convention
    applied to static structures.
    """
    universe = handle.universe
    atoms_a = universe.select_atoms(f"({selection_a}) and (name N* or name O* or name S*)")
    atoms_b = universe.select_atoms(f"({selection_b}) and (name N* or name O* or name S*)")
    rows = []
    for ts in universe.trajectory[:: handle.stride]:
        if len(atoms_a) == 0 or len(atoms_b) == 0:
            rows.append({"frame": ts.frame, "time_ps": float(ts.time), "n_hydrogen_bonds": 0})
            continue
        tree_b = cKDTree(atoms_b.positions)
        pairs = cKDTree(atoms_a.positions).query_ball_tree(tree_b, r=distance_cutoff)
        n_bonds = sum(len(neighbours) for neighbours in pairs)
        rows.append({"frame": ts.frame, "time_ps": float(ts.time), "n_hydrogen_bonds": n_bonds})
    return pd.DataFrame(rows)
