"""Per-frame feature extraction for the temporal MD model (section 26).

Aggregates RMSD, contact count, COM distance and hydrogen-bond count into
a single per-frame feature table that :mod:`enhancoai.models.temporal`
consumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from enhancoai.md.loader import TrajectoryHandle
from enhancoai.md.contacts import center_of_mass_distance, hydrogen_bond_count_trajectory
from enhancoai.md.rmsd import compute_rmsd

FEATURE_COLUMNS = ["rmsd_protein", "rmsd_dna", "com_distance", "n_hydrogen_bonds"]


def extract_frame_features(
    handle: TrajectoryHandle,
    protein_selection: str = "protein and name CA",
    dna_selection: str = "nucleic",
) -> pd.DataFrame:
    """Build a (n_frames, n_features) table suitable for the temporal model."""
    rmsd_protein = compute_rmsd(handle, selection=protein_selection)
    rmsd_dna = compute_rmsd(handle, selection=dna_selection)
    com = center_of_mass_distance(handle, protein_selection, dna_selection)
    hbonds = hydrogen_bond_count_trajectory(handle, protein_selection, dna_selection)

    merged = rmsd_protein.rename(columns={"rmsd": "rmsd_protein"})[["frame", "time_ps", "rmsd_protein"]]
    merged = merged.merge(rmsd_dna.rename(columns={"rmsd": "rmsd_dna"})[["frame", "rmsd_dna"]], on="frame")
    merged = merged.merge(com[["frame", "com_distance"]], on="frame")
    merged = merged.merge(hbonds[["frame", "n_hydrogen_bonds"]], on="frame")
    return merged


def to_feature_tensor(features: pd.DataFrame):
    """Convert the per-frame feature table to a (n_frames, n_features) torch.Tensor."""
    import torch

    array = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    return torch.from_numpy(array)
