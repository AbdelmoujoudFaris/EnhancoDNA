"""General-purpose 2D plots: RMSD/RMSF traces, PMF curves, training curves."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from enhancoai.free_energy.pmf import PMFResult


def plot_timeseries(frame: pd.DataFrame, x: str, y: str, title: str | None = None, ylabel: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(frame[x], frame[y])
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title or y)
    fig.tight_layout()
    return fig


def plot_rmsf(rmsf: pd.DataFrame, title: str = "RMSF"):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(rmsf["res_seq"], rmsf["rmsf"])
    ax.set_xlabel("Residue")
    ax.set_ylabel("RMSF (A)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pmf(pmf: PMFResult, title: str = "Potential of mean force"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pmf.bin_centers, pmf.free_energy, marker="o", markersize=3)
    ax.set_xlabel("Reaction coordinate")
    ax.set_ylabel(f"Free energy (kcal/mol), T={pmf.temperature_k} K")
    subtitle = f"method: {pmf.method}, n={pmf.n_samples}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_training_curves(history: list[dict]):
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    if "train_total" in frame:
        axes[0].plot(frame["epoch"], frame["train_total"], label="train")
    if "val_total" in frame:
        axes[0].plot(frame["epoch"], frame["val_total"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    for metric in ("val_f1", "val_auroc", "val_mcc"):
        if metric in frame:
            axes[1].plot(frame["epoch"], frame[metric], label=metric)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    return fig
