"""Potential of mean force (PMF) estimation from equilibrium samples.

ΔG(x) = -kB T ln P(x) + C

This module computes PMFs from the *observed* histogram of a reaction
coordinate. This is only a valid free-energy estimate if the input samples
come from an equilibrium (unbiased, adequately converged) simulation. It
is NOT a substitute for umbrella sampling / WHAM / metadynamics reweighting
when the underlying sampling is biased or the barrier is high relative to
kT sampled directly. Every PMF produced by this module is tagged with its
``method`` field so results are never silently over-interpreted; see
section 17 of the project specification and ``docs/`` for details.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BOLTZMANN_KCAL_PER_MOL_K = 1.987204e-3  # kcal / (mol K)


@dataclass
class PMFResult:
    bin_centers: np.ndarray
    free_energy: np.ndarray  # kcal/mol, relative to the minimum
    counts: np.ndarray
    temperature_k: float
    method: str
    n_samples: int
    warnings: list[str] = field(default_factory=list)


def compute_pmf_1d(
    samples: np.ndarray,
    n_bins: int = 50,
    temperature_k: float = 300.0,
    min_count_for_estimate: int = 5,
) -> PMFResult:
    """1D PMF from a histogram of an equilibrium reaction-coordinate sample.

    Bins with fewer than ``min_count_for_estimate`` samples are set to NaN
    rather than being extrapolated, since a raw histogram there is not a
    statistically meaningful free-energy estimate.
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]
    warnings: list[str] = []
    if len(samples) < 50:
        warnings.append(
            f"Only {len(samples)} samples provided; PMF estimate is likely unconverged. "
            "Treat as exploratory, not a converged free-energy profile."
        )

    counts, edges = np.histogram(samples, bins=n_bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    kbt = BOLTZMANN_KCAL_PER_MOL_K * temperature_k
    prob = counts / max(counts.sum(), 1)
    with np.errstate(divide="ignore"):
        free_energy = -kbt * np.log(np.where(prob > 0, prob, np.nan))
    free_energy = free_energy - np.nanmin(free_energy)
    free_energy = np.where(counts >= min_count_for_estimate, free_energy, np.nan)

    if np.isnan(free_energy).any():
        warnings.append(
            "Some bins had fewer than the minimum sample count and are reported as NaN "
            "rather than extrapolated."
        )

    return PMFResult(
        bin_centers=bin_centers,
        free_energy=free_energy,
        counts=counts,
        temperature_k=temperature_k,
        method="equilibrium_histogram (raw, unbiased-sampling assumption)",
        n_samples=len(samples),
        warnings=warnings,
    )


def compute_pmf_2d(
    samples_x: np.ndarray,
    samples_y: np.ndarray,
    n_bins: int = 30,
    temperature_k: float = 300.0,
    min_count_for_estimate: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D PMF surface; returns (x_edges, y_edges, free_energy_grid) in kcal/mol."""
    samples_x = np.asarray(samples_x, dtype=float)
    samples_y = np.asarray(samples_y, dtype=float)
    mask = ~(np.isnan(samples_x) | np.isnan(samples_y))
    samples_x, samples_y = samples_x[mask], samples_y[mask]

    counts, x_edges, y_edges = np.histogram2d(samples_x, samples_y, bins=n_bins)
    kbt = BOLTZMANN_KCAL_PER_MOL_K * temperature_k
    prob = counts / max(counts.sum(), 1)
    with np.errstate(divide="ignore"):
        free_energy = -kbt * np.log(np.where(prob > 0, prob, np.nan))
    free_energy = free_energy - np.nanmin(free_energy)
    free_energy = np.where(counts >= min_count_for_estimate, free_energy, np.nan)
    return x_edges, y_edges, free_energy


def free_energy_at_extremes(pmf: PMFResult, low_fraction: float = 0.2, high_fraction: float = 0.2) -> tuple[float, float]:
    """Count-weighted mean free energy in the lowest- and highest-coordinate bins.

    Used to estimate a binding free energy from a PMF when the reaction
    coordinate has a clear bound/unbound end (e.g. a protein-DNA distance,
    where low = bound and high = unbound). Returns (mean_low, mean_high),
    each NaN if no valid (sufficiently sampled) bins fall in that range.
    """
    order = np.argsort(pmf.bin_centers)
    centers = pmf.bin_centers[order]
    energy = pmf.free_energy[order]
    counts = pmf.counts[order]

    n_low = max(1, int(round(len(centers) * low_fraction)))
    n_high = max(1, int(round(len(centers) * high_fraction)))

    def _weighted_mean(e: np.ndarray, c: np.ndarray) -> float:
        valid = ~np.isnan(e)
        if not valid.any() or c[valid].sum() == 0:
            return float("nan")
        return float(np.average(e[valid], weights=c[valid]))

    mean_low = _weighted_mean(energy[:n_low], counts[:n_low])
    mean_high = _weighted_mean(energy[-n_high:], counts[-n_high:])
    return mean_low, mean_high


def pmf_to_frame(result: PMFResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_coordinate": result.bin_centers,
            "free_energy_kcal_mol": result.free_energy,
            "counts": result.counts,
        }
    )
