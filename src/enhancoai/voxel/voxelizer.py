"""Convert a protein-DNA interface into a multi-channel 3D voxel grid.

Atoms are splatted onto the grid with a Gaussian kernel (per-atom sigma
proportional to a nominal van der Waals radius) rather than hard binary
occupancy, which keeps the representation differentiable-friendly and
robust to small coordinate/grid-alignment differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from enhancoai.voxel.representation import (
    N_CHANNELS,
    CHANNEL_NAMES,
    PARTIAL_CHARGE,
    HYDROPHOBICITY,
    DONOR_ATOMS,
    ACCEPTOR_ATOMS,
    DNA_BACKBONE_ATOMS,
)


@dataclass
class VoxelGrid:
    grid: np.ndarray  # (N_CHANNELS, size, size, size)
    center: np.ndarray
    voxel_size: float
    grid_size: int
    channel_names: list[str]


def _splat(grid: np.ndarray, channel: int, voxel_coords: np.ndarray, values: np.ndarray, sigma_voxels: float, grid_size: int) -> None:
    """Add a Gaussian-weighted contribution of each atom to its neighbouring voxels."""
    radius = max(1, int(np.ceil(3 * sigma_voxels)))
    for (vx, vy, vz), value in zip(voxel_coords, values):
        if value == 0:
            continue
        x0, x1 = max(0, vx - radius), min(grid_size, vx + radius + 1)
        y0, y1 = max(0, vy - radius), min(grid_size, vy + radius + 1)
        z0, z1 = max(0, vz - radius), min(grid_size, vz + radius + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue
        xs, ys, zs = np.meshgrid(
            np.arange(x0, x1), np.arange(y0, y1), np.arange(z0, z1), indexing="ij"
        )
        dist2 = (xs - vx) ** 2 + (ys - vy) ** 2 + (zs - vz) ** 2
        weight = np.exp(-dist2 / (2 * sigma_voxels**2))
        grid[channel, x0:x1, y0:y1, z0:z1] += value * weight


def voxelize_interface(
    protein_atoms: pd.DataFrame,
    dna_atoms: pd.DataFrame,
    center: np.ndarray | None = None,
    grid_size: int = 32,
    voxel_size: float = 1.0,
    atom_sigma_angstrom: float = 1.0,
) -> VoxelGrid:
    """Build the (N_CHANNELS, grid_size, grid_size, grid_size) interface voxel grid.

    Channels follow :data:`enhancoai.voxel.representation.CHANNEL_NAMES`.
    """
    all_coords = pd.concat(
        [protein_atoms[["x", "y", "z"]], dna_atoms[["x", "y", "z"]]], ignore_index=True
    ).to_numpy(dtype=float)
    if center is None:
        center = all_coords.mean(axis=0) if len(all_coords) else np.zeros(3)

    grid = np.zeros((N_CHANNELS, grid_size, grid_size, grid_size), dtype=np.float32)
    sigma_voxels = max(atom_sigma_angstrom / voxel_size, 0.5)
    half = grid_size / 2.0

    def _to_voxel(coords: np.ndarray) -> np.ndarray:
        relative = (coords - center) / voxel_size + half
        return np.round(relative).astype(int)

    def _in_bounds(voxel_coords: np.ndarray) -> np.ndarray:
        return np.all((voxel_coords >= 0) & (voxel_coords < grid_size), axis=1)

    idx = {name: i for i, name in enumerate(CHANNEL_NAMES)}

    if len(protein_atoms):
        coords = protein_atoms[["x", "y", "z"]].to_numpy(dtype=float)
        voxels = _to_voxel(coords)
        mask = _in_bounds(voxels)
        voxels, sel = voxels[mask], protein_atoms[mask]

        _splat(grid, idx["protein_occupancy"], voxels, np.ones(len(voxels)), sigma_voxels, grid_size)

        charges = sel["atom_name"].str.strip().map(PARTIAL_CHARGE).fillna(0.0).to_numpy()
        _splat(grid, idx["electrostatic_potential"], voxels, charges, sigma_voxels, grid_size)

        hydro = sel["res_name"].str.strip().map(HYDROPHOBICITY).fillna(0.0).to_numpy()
        _splat(grid, idx["hydrophobicity"], voxels, hydro, sigma_voxels, grid_size)

        donor = sel["atom_name"].str.strip().isin(DONOR_ATOMS).astype(float).to_numpy()
        _splat(grid, idx["hydrogen_bond_donor"], voxels, donor, sigma_voxels, grid_size)

        acceptor = sel["atom_name"].str.strip().isin(ACCEPTOR_ATOMS).astype(float).to_numpy()
        _splat(grid, idx["hydrogen_bond_acceptor"], voxels, acceptor, sigma_voxels, grid_size)

    if len(dna_atoms):
        coords = dna_atoms[["x", "y", "z"]].to_numpy(dtype=float)
        voxels = _to_voxel(coords)
        mask = _in_bounds(voxels)
        voxels, sel = voxels[mask], dna_atoms[mask]

        _splat(grid, idx["dna_occupancy"], voxels, np.ones(len(voxels)), sigma_voxels, grid_size)

        charges = sel["atom_name"].str.strip().map(PARTIAL_CHARGE).fillna(0.0).to_numpy()
        _splat(grid, idx["electrostatic_potential"], voxels, charges, sigma_voxels, grid_size)

        backbone = sel["atom_name"].str.strip().isin(DNA_BACKBONE_ATOMS).astype(float).to_numpy()
        _splat(grid, idx["dna_backbone"], voxels, backbone, sigma_voxels, grid_size)

        base_id = (~sel["atom_name"].str.strip().isin(DNA_BACKBONE_ATOMS)).astype(float).to_numpy()
        _splat(grid, idx["base_identity"], voxels, base_id, sigma_voxels, grid_size)

    grid[idx["interface_density"]] = grid[idx["protein_occupancy"]] * grid[idx["dna_occupancy"]]

    xs, ys, zs = np.meshgrid(
        np.arange(grid_size), np.arange(grid_size), np.arange(grid_size), indexing="ij"
    )
    grid[idx["distance_field"]] = np.sqrt(
        (xs - half) ** 2 + (ys - half) ** 2 + (zs - half) ** 2
    ) * voxel_size

    return VoxelGrid(
        grid=grid, center=center, voxel_size=voxel_size, grid_size=grid_size, channel_names=CHANNEL_NAMES
    )


def to_tensor(voxel_grid: VoxelGrid):
    import torch

    return torch.from_numpy(voxel_grid.grid).unsqueeze(0)  # (1, C, D, H, W) - batch dim of 1
