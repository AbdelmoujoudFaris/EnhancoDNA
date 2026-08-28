"""DNA global/local curvature from the sequence of base-pair origins."""

from __future__ import annotations

import numpy as np
import pandas as pd

from enhancoai.dna.geometry import BasePairFrame


def local_curvature(frames: list[BasePairFrame], window: int = 3) -> pd.DataFrame:
    """Bending angle (degrees) between the helical axis segments before/after
    each base pair, averaged over ``window`` steps on each side.

    A larger angle indicates sharper local DNA bending at that position.
    """
    origins = np.array([f.origin for f in frames])
    n = len(origins)
    rows = []
    for i in range(n):
        lo, hi = max(0, i - window), min(n - 1, i + window)
        if lo == i or hi == i:
            rows.append({"index": i, "res_seq_a": frames[i].res_seq_a, "curvature_deg": np.nan})
            continue
        v1 = origins[i] - origins[lo]
        v2 = origins[hi] - origins[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            angle = np.nan
        else:
            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
        rows.append({"index": i, "res_seq_a": frames[i].res_seq_a, "curvature_deg": angle})
    return pd.DataFrame(rows)


def global_bend_angle(frames: list[BasePairFrame]) -> float:
    """Overall bending angle: angle between the first and last local helical tangents."""
    if len(frames) < 2:
        return 0.0
    v1 = frames[0].z_axis
    v2 = frames[-1].z_axis
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def end_to_end_contour_ratio(frames: list[BasePairFrame]) -> float:
    """Ratio of end-to-end distance to contour (arc) length; 1.0 = perfectly straight."""
    origins = np.array([f.origin for f in frames])
    if len(origins) < 2:
        return 1.0
    end_to_end = np.linalg.norm(origins[-1] - origins[0])
    contour = np.sum(np.linalg.norm(np.diff(origins, axis=0), axis=1))
    if contour < 1e-6:
        return 1.0
    return float(end_to_end / contour)
