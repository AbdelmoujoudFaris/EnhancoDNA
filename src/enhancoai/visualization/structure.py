"""3D structure rendering: protein, DNA, contacts, allosteric network, AI attribution overlays."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enhancoai.visualization import HAS_PYVISTA

CHAIN_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def render_structure_matplotlib(structure, chain_ids: list[str] | None = None, title: str = "Structure"):
    """Static 3D scatter of atom positions coloured by chain (matplotlib fallback)."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    chain_ids = chain_ids or structure.chain_ids
    for i, chain_id in enumerate(chain_ids):
        coords = structure.chain(chain_id)[["x", "y", "z"]].to_numpy(dtype=float)
        if len(coords) == 0:
            continue
        ax.scatter(*coords.T, s=4, color=CHAIN_COLORS[i % len(CHAIN_COLORS)], label=f"Chain {chain_id}")

    ax.set_title(title)
    ax.set_xlabel("X (A)")
    ax.set_ylabel("Y (A)")
    ax.set_zlabel("Z (A)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def render_structure_pyvista(structure, chain_ids: list[str] | None = None):
    """Interactive PyVista plotter, only usable when PyVista is installed."""
    if not HAS_PYVISTA:
        raise ImportError("PyVista is not installed; use render_structure_matplotlib() instead.")
    import pyvista as pv

    plotter = pv.Plotter()
    chain_ids = chain_ids or structure.chain_ids
    for i, chain_id in enumerate(chain_ids):
        coords = structure.chain(chain_id)[["x", "y", "z"]].to_numpy(dtype=float)
        if len(coords) == 0:
            continue
        cloud = pv.PolyData(coords)
        plotter.add_mesh(cloud, color=CHAIN_COLORS[i % len(CHAIN_COLORS)], point_size=6, render_points_as_spheres=True, label=f"Chain {chain_id}")
    plotter.add_legend()
    return plotter


def render_attribution_overlay(structure, importance_frame, title: str = "AI Attribution"):
    """Colour atoms by an importance score (0-1) from low (blue) to high (red)."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    merged = importance_frame.set_index(["chain_id", "res_seq"])
    coords, colors = [], []
    for chain_id in structure.chain_ids:
        chain_atoms = structure.chain(chain_id)
        for res_seq, group in chain_atoms.groupby("res_seq"):
            key = (chain_id, res_seq)
            if key not in merged.index:
                continue
            score = merged.loc[key, "importance_normalised"]
            centroid = group[["x", "y", "z"]].to_numpy(dtype=float).mean(axis=0)
            coords.append(centroid)
            colors.append(score)

    if coords:
        coords = np.array(coords)
        scatter = ax.scatter(*coords.T, c=colors, cmap="coolwarm", s=20)
        fig.colorbar(scatter, ax=ax, shrink=0.6, label="Importance")

    ax.set_title(title)
    fig.tight_layout()
    return fig
