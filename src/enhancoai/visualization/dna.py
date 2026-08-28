"""DNA-specific visualisation: ribbon trace, curvature, groove width, AI score mapping."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enhancoai.dna.geometry import BasePairFrame


def render_dna_ribbon(frames: list[BasePairFrame], color_by: np.ndarray | None = None, title: str = "DNA axis"):
    """3D line trace of the base-pair origin sequence (a simplified DNA "ribbon"),
    optionally colour-mapped by a per-base-pair scalar (e.g. curvature or AI score).
    """
    origins = np.array([f.origin for f in frames])
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    if color_by is not None and len(color_by) == len(origins):
        scatter = ax.scatter(*origins.T, c=color_by, cmap="viridis", s=15)
        fig.colorbar(scatter, ax=ax, shrink=0.6)
    ax.plot(*origins.T, color="grey", linewidth=1, alpha=0.6)

    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_parameter_vs_position(step_params, column: str, title: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(step_params["step_index"], step_params[column], marker="o", markersize=3)
    ax.set_xlabel("Base-pair step index (DNA position)")
    ax.set_ylabel(column)
    ax.set_title(title or f"{column} vs DNA position")
    fig.tight_layout()
    return fig
