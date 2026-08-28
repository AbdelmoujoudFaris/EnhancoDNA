"""Reaction-coordinate extraction for protein-DNA (un)binding analysis (section 16)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from enhancoai.md.loader import TrajectoryHandle
from enhancoai.md.contacts import center_of_mass_distance, hydrogen_bond_count_trajectory
from enhancoai.md.orientation import orientation_trajectory


def build_reaction_coordinates(
    handle: TrajectoryHandle,
    protein_selection: str,
    dna_selection: str,
    native_contact_pairs: list[tuple[int, int]] | None = None,
    native_contact_cutoff: float = 8.0,
) -> pd.DataFrame:
    """Assemble a multidimensional reaction-coordinate table for one trajectory.

    Columns: com_distance, n_hydrogen_bonds, orientation_deg, and
    (if ``native_contact_pairs`` given) native_contact_fraction -- the
    fraction of a reference ("bound-state") contact set still intact in
    each frame.
    """
    com = center_of_mass_distance(handle, protein_selection, dna_selection)
    hbonds = hydrogen_bond_count_trajectory(handle, protein_selection, dna_selection)
    orientation = orientation_trajectory(handle, protein_selection, dna_selection)

    merged = com.merge(hbonds[["frame", "n_hydrogen_bonds"]], on="frame")
    merged = merged.merge(orientation[["frame", "angle_deg"]], on="frame")

    if native_contact_pairs:
        merged["native_contact_fraction"] = _native_contact_fraction(
            handle, native_contact_pairs, native_contact_cutoff
        )

    return merged


def _native_contact_fraction(
    handle: TrajectoryHandle,
    native_contact_pairs: list[tuple[int, int]],
    cutoff: float,
) -> list[float]:
    universe = handle.universe
    fractions = []
    for ts in universe.trajectory[:: handle.stride]:
        intact = 0
        for resid_a, resid_b in native_contact_pairs:
            atoms_a = universe.select_atoms(f"resid {resid_a}")
            atoms_b = universe.select_atoms(f"resid {resid_b}")
            if len(atoms_a) == 0 or len(atoms_b) == 0:
                continue
            com_a, com_b = atoms_a.center_of_mass(), atoms_b.center_of_mass()
            if np.linalg.norm(com_a - com_b) <= cutoff:
                intact += 1
        fractions.append(intact / max(len(native_contact_pairs), 1))
    return fractions


def classify_binding_state(
    com_distance: float,
    n_hydrogen_bonds: int,
    bound_distance_threshold: float = 15.0,
    detached_distance_threshold: float = 30.0,
    bound_hbond_threshold: int = 3,
) -> str:
    """Simple threshold-based classification: bound / partially_bound / detached."""
    if com_distance <= bound_distance_threshold and n_hydrogen_bonds >= bound_hbond_threshold:
        return "bound"
    if com_distance >= detached_distance_threshold:
        return "detached"
    return "partially_bound"
