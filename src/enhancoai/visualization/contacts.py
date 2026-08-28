"""Interactive-ready contact map rendering (protein-DNA and protein-protein)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_contact_map(residue_contact_map: pd.DataFrame, title: str = "Contact map"):
    """Heatmap of min_distance for a residue_contact_map DataFrame."""
    if residue_contact_map.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No contacts found", ha="center", va="center")
        ax.axis("off")
        return fig

    rows = sorted(residue_contact_map["res_seq_a"].unique())
    cols = sorted(residue_contact_map["res_seq_b"].unique())
    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: i for i, c in enumerate(cols)}

    matrix = np.full((len(rows), len(cols)), np.nan)
    for record in residue_contact_map.itertuples():
        matrix[row_index[record.res_seq_a], col_index[record.res_seq_b]] = record.min_distance

    fig, ax = plt.subplots(figsize=(max(4, len(cols) * 0.3), max(3, len(rows) * 0.3)))
    im = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6)
    ax.set_xlabel("Chain B residue")
    ax.set_ylabel("Chain A residue")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Min. distance (A)", shrink=0.7)
    fig.tight_layout()
    return fig


def render_correlation_matrix(correlation_matrix: pd.DataFrame, title: str = "Dynamic cross-correlation"):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(correlation_matrix.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    return fig
