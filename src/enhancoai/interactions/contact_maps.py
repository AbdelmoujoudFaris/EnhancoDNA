"""Generic distance-based contact detection and contact-map construction.

All contact detection in EnhancoAI reduces to pairwise distance queries
between two atom selections (protein-DNA, protein-protein, or an arbitrary
pair of chains). This module provides the shared, tested primitive that the
more specific analyses in :mod:`enhancoai.interactions` build on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass
class Contact:
    chain_a: str
    res_seq_a: int
    res_name_a: str
    atom_a: str
    chain_b: str
    res_seq_b: int
    res_name_b: str
    atom_b: str
    distance: float


def _atom_group_arrays(atoms: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    coords = atoms[["x", "y", "z"]].to_numpy(dtype=float)
    meta = atoms[["chain_id", "res_seq", "res_name", "atom_name"]].reset_index(drop=True)
    return coords, meta


def find_atom_contacts(
    atoms_a: pd.DataFrame,
    atoms_b: pd.DataFrame,
    cutoff: float = 5.0,
) -> list[Contact]:
    """Find all atom-atom pairs between two atom sets within ``cutoff`` angstrom.

    Uses a KD-tree so it scales to full structures rather than O(n*m) loops.
    """
    if len(atoms_a) == 0 or len(atoms_b) == 0:
        return []

    coords_a, meta_a = _atom_group_arrays(atoms_a)
    coords_b, meta_b = _atom_group_arrays(atoms_b)

    tree_b = cKDTree(coords_b)
    pairs = cKDTree(coords_a).query_ball_tree(tree_b, r=cutoff)

    contacts: list[Contact] = []
    for i, neighbours in enumerate(pairs):
        if not neighbours:
            continue
        row_a = meta_a.iloc[i]
        for j in neighbours:
            row_b = meta_b.iloc[j]
            dist = float(np.linalg.norm(coords_a[i] - coords_b[j]))
            contacts.append(
                Contact(
                    chain_a=row_a.chain_id,
                    res_seq_a=int(row_a.res_seq),
                    res_name_a=row_a.res_name,
                    atom_a=row_a.atom_name,
                    chain_b=row_b.chain_id,
                    res_seq_b=int(row_b.res_seq),
                    res_name_b=row_b.res_name,
                    atom_b=row_b.atom_name,
                    distance=dist,
                )
            )
    return contacts


def contacts_to_frame(contacts: list[Contact]) -> pd.DataFrame:
    if not contacts:
        return pd.DataFrame(
            columns=[
                "chain_a", "res_seq_a", "res_name_a", "atom_a",
                "chain_b", "res_seq_b", "res_name_b", "atom_b", "distance",
            ]
        )
    return pd.DataFrame([c.__dict__ for c in contacts])


def residue_contact_map(contacts: list[Contact]) -> pd.DataFrame:
    """Collapse atom-atom contacts into a residue-residue contact map.

    Each row is a unique (chain_a, res_seq_a) <-> (chain_b, res_seq_b) pair
    with the minimum observed atom-atom distance and the number of
    contributing atom-atom contacts.
    """
    frame = contacts_to_frame(contacts)
    if frame.empty:
        return frame.assign(n_atom_contacts=[])
    grouped = (
        frame.groupby(["chain_a", "res_seq_a", "res_name_a", "chain_b", "res_seq_b", "res_name_b"])
        .agg(min_distance=("distance", "min"), n_atom_contacts=("distance", "size"))
        .reset_index()
        .sort_values("min_distance")
    )
    return grouped


def contact_frequency(per_frame_contacts: list[list[Contact]]) -> pd.DataFrame:
    """Compute contact persistence (fraction of frames in which each residue
    pair is in contact) across a set of frames (e.g. an MD trajectory)."""
    n_frames = len(per_frame_contacts)
    if n_frames == 0:
        return pd.DataFrame(columns=["chain_a", "res_seq_a", "chain_b", "res_seq_b", "frequency"])

    counts: dict[tuple, int] = {}
    for frame_contacts in per_frame_contacts:
        seen = set()
        for c in frame_contacts:
            key = (c.chain_a, c.res_seq_a, c.chain_b, c.res_seq_b)
            seen.add(key)
        for key in seen:
            counts[key] = counts.get(key, 0) + 1

    rows = [
        {
            "chain_a": key[0],
            "res_seq_a": key[1],
            "chain_b": key[2],
            "res_seq_b": key[3],
            "frequency": count / n_frames,
        }
        for key, count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values("frequency", ascending=False).reset_index(drop=True)
