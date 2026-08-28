"""DNA geometric descriptor features -- feed the Allosteric Fingerprint (section 21)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from enhancoai.dna.curvature import global_bend_angle, end_to_end_contour_ratio
from enhancoai.dna.geometry import BasePairFrame


def dna_geometry_summary(
    frames: list[BasePairFrame],
    step_params: pd.DataFrame,
    groove: pd.DataFrame,
) -> dict[str, float]:
    summary = {
        "global_bend_deg": global_bend_angle(frames),
        "end_to_end_contour_ratio": end_to_end_contour_ratio(frames),
    }
    for col in ("twist", "roll", "tilt", "rise", "slide", "shift"):
        if col in step_params:
            summary[f"{col}_mean"] = float(step_params[col].mean())
            summary[f"{col}_std"] = float(step_params[col].std())
    for col in ("minor_groove_width", "major_groove_width"):
        if col in groove:
            summary[f"{col}_mean"] = float(np.nanmean(groove[col]))
            summary[f"{col}_std"] = float(np.nanstd(groove[col]))
    return summary
