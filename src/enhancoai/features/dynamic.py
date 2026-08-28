"""Summary statistics of MD-derived dynamic features (RMSF, correlation, RMSD)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmsf_summary(rmsf: pd.DataFrame) -> dict[str, float]:
    values = rmsf["rmsf"].to_numpy(dtype=float)
    return {
        "rmsf_mean": float(np.mean(values)),
        "rmsf_std": float(np.std(values)),
        "rmsf_max": float(np.max(values)),
    }


def rmsd_summary(rmsd: pd.DataFrame) -> dict[str, float]:
    values = rmsd["rmsd"].to_numpy(dtype=float)
    return {
        "rmsd_mean": float(np.mean(values)),
        "rmsd_std": float(np.std(values)),
        "rmsd_final": float(values[-1]) if len(values) else float("nan"),
    }


def correlation_summary(correlation_matrix: pd.DataFrame) -> dict[str, float]:
    values = correlation_matrix.to_numpy(dtype=float)
    off_diag = values[~np.eye(values.shape[0], dtype=bool)]
    return {
        "mean_abs_correlation": float(np.mean(np.abs(off_diag))) if len(off_diag) else float("nan"),
        "max_abs_correlation": float(np.max(np.abs(off_diag))) if len(off_diag) else float("nan"),
    }
