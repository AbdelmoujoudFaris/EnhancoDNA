"""Base-pair detection and simplified local reference frames.

Watson-Crick pairing is detected geometrically (C1'-C1' distance close to
the canonical ~10.4 A, plus proximity of the Watson-Crick edge atoms),
which is robust to sequence but is a geometric proxy, not a hydrogen-bond
based pairing assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PURINES = {"DA", "DG", "A", "G"}
PYRIMIDINES = {"DT", "DC", "DU", "T", "C", "U"}

# Watson-Crick edge atom used per base type to check pairing proximity.
WC_EDGE_ATOM = {
    "DA": "N1", "A": "N1",
    "DG": "N1", "G": "N1",
    "DT": "N3", "T": "N3", "DU": "N3", "U": "N3",
    "DC": "N3", "C": "N3",
}

C1_ATOM = "C1'"


@dataclass
class BasePairFrame:
    index: int
    res_seq_a: int
    res_seq_b: int
    res_name_a: str
    res_name_b: str
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray


def _residue_atom(chain_atoms: pd.DataFrame, res_seq: int, atom_name: str) -> np.ndarray | None:
    match = chain_atoms[(chain_atoms["res_seq"] == res_seq) & (chain_atoms["atom_name"].str.strip() == atom_name)]
    if match.empty:
        return None
    return match[["x", "y", "z"]].to_numpy(dtype=float)[0]


def find_watson_crick_pairs(
    structure,
    chain_a: str,
    chain_b: str,
    c1_distance_range: tuple[float, float] = (9.0, 11.5),
    edge_distance_max: float = 3.5,
) -> list[tuple[int, int]]:
    """Geometrically pair residues of two DNA strands.

    Returns a list of (res_seq_a, res_seq_b) pairs, ordered along chain_a.
    """
    a_atoms = structure.chain(chain_a)
    b_atoms = structure.chain(chain_b)

    a_residues = a_atoms.drop_duplicates(subset=["res_seq"])[["res_seq", "res_name"]].values
    b_residues = b_atoms.drop_duplicates(subset=["res_seq"])[["res_seq", "res_name"]].values

    pairs: list[tuple[int, int]] = []
    used_b = set()
    for res_seq_a, res_name_a in a_residues:
        c1_a = _residue_atom(a_atoms, res_seq_a, C1_ATOM)
        edge_atom_name = WC_EDGE_ATOM.get(res_name_a.strip())
        edge_a = _residue_atom(a_atoms, res_seq_a, edge_atom_name) if edge_atom_name else None
        if c1_a is None or edge_a is None:
            continue

        best_match, best_dist = None, np.inf
        for res_seq_b, res_name_b in b_residues:
            if res_seq_b in used_b:
                continue
            c1_b = _residue_atom(b_atoms, res_seq_b, C1_ATOM)
            edge_atom_name_b = WC_EDGE_ATOM.get(res_name_b.strip())
            edge_b = _residue_atom(b_atoms, res_seq_b, edge_atom_name_b) if edge_atom_name_b else None
            if c1_b is None or edge_b is None:
                continue
            c1_dist = np.linalg.norm(c1_a - c1_b)
            if not (c1_distance_range[0] <= c1_dist <= c1_distance_range[1]):
                continue
            edge_dist = np.linalg.norm(edge_a - edge_b)
            if edge_dist > edge_distance_max:
                continue
            if edge_dist < best_dist:
                best_match, best_dist = res_seq_b, edge_dist

        if best_match is not None:
            pairs.append((int(res_seq_a), int(best_match)))
            used_b.add(best_match)

    return pairs


def base_pair_frames(structure, chain_a: str, chain_b: str, pairs: list[tuple[int, int]]) -> list[BasePairFrame]:
    """Construct a simplified local reference frame for each base pair.

    origin: midpoint of the two C1' atoms.
    x-axis: unit vector from strand-b C1' to strand-a C1' (short/long axis proxy).
    z-axis: local helical tangent estimated from neighbouring base-pair
        origins (central difference; forward/backward difference at the ends).
    y-axis: completes a right-handed orthonormal frame (z cross x).
    """
    a_atoms = structure.chain(chain_a)
    b_atoms = structure.chain(chain_b)

    origins = []
    x_axes = []
    meta = []
    for res_seq_a, res_seq_b in pairs:
        c1_a = _residue_atom(a_atoms, res_seq_a, C1_ATOM)
        c1_b = _residue_atom(b_atoms, res_seq_b, C1_ATOM)
        if c1_a is None or c1_b is None:
            continue
        origin = (c1_a + c1_b) / 2.0
        x_axis = c1_a - c1_b
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
        origins.append(origin)
        x_axes.append(x_axis)
        res_name_a = a_atoms[a_atoms["res_seq"] == res_seq_a]["res_name"].iloc[0]
        res_name_b = b_atoms[b_atoms["res_seq"] == res_seq_b]["res_name"].iloc[0]
        meta.append((res_seq_a, res_seq_b, res_name_a, res_name_b))

    n = len(origins)
    frames: list[BasePairFrame] = []
    for i in range(n):
        if n == 1:
            tangent = np.array([0.0, 0.0, 1.0])
        elif i == 0:
            tangent = origins[1] - origins[0]
        elif i == n - 1:
            tangent = origins[i] - origins[i - 1]
        else:
            tangent = origins[i + 1] - origins[i - 1]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)

        x_axis = x_axes[i]
        # orthogonalise x against tangent (Gram-Schmidt) to keep a right-handed frame
        x_axis = x_axis - np.dot(x_axis, tangent) * tangent
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
        z_axis = tangent
        y_axis = np.cross(z_axis, x_axis)

        res_seq_a, res_seq_b, res_name_a, res_name_b = meta[i]
        frames.append(
            BasePairFrame(
                index=i,
                res_seq_a=res_seq_a,
                res_seq_b=res_seq_b,
                res_name_a=res_name_a,
                res_name_b=res_name_b,
                origin=origins[i],
                x_axis=x_axis,
                y_axis=y_axis,
                z_axis=z_axis,
            )
        )
    return frames
