"""Simplified base-pair step parameters: twist, roll, tilt, rise, slide, shift.

Computed from consecutive :class:`~enhancoai.dna.geometry.BasePairFrame`
objects using the mid-frame convention: the "mean frame" between step i and
i+1 is used to resolve translational parameters (shift, slide, rise) and
the rotation between the two frames is decomposed into tilt/roll/twist.
This mirrors the spirit of the standard local base-pair-step definition
but is a simplified reimplementation -- see module-level docstring in
:mod:`enhancoai.dna`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from enhancoai.dna.geometry import BasePairFrame


def _rotation_matrix(frame: BasePairFrame) -> np.ndarray:
    return np.stack([frame.x_axis, frame.y_axis, frame.z_axis], axis=1)


def _mean_frame_rotation(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Average two rotation matrices via the axis-angle of their relative rotation (half-angle)."""
    relative = r1.T @ r2
    # Axis-angle of relative rotation
    angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0))
    if np.isclose(angle, 0.0):
        return r1
    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = angle / 2.0
    k = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    half_rotation = np.eye(3) + np.sin(half) * k + (1 - np.cos(half)) * (k @ k)
    return r1 @ half_rotation


def step_parameters(frames: list[BasePairFrame]) -> pd.DataFrame:
    """Compute shift, slide, rise, tilt, roll, twist for each consecutive base-pair step."""
    rows = []
    for i in range(len(frames) - 1):
        f1, f2 = frames[i], frames[i + 1]
        r1, r2 = _rotation_matrix(f1), _rotation_matrix(f2)
        r_mean = _mean_frame_rotation(r1, r2)

        d_origin = f2.origin - f1.origin
        shift, slide, rise = r_mean.T @ d_origin

        relative = r1.T @ r2
        tilt = np.degrees(np.arctan2(relative[2, 1], relative[2, 2]))
        roll = np.degrees(np.arctan2(-relative[2, 0], np.sqrt(relative[2, 1] ** 2 + relative[2, 2] ** 2)))
        twist = np.degrees(np.arctan2(relative[1, 0], relative[0, 0]))

        rows.append(
            {
                "step_index": i,
                "res_seq_a_1": f1.res_seq_a,
                "res_seq_a_2": f2.res_seq_a,
                "shift": float(shift),
                "slide": float(slide),
                "rise": float(rise),
                "tilt": float(tilt),
                "roll": float(roll),
                "twist": float(twist),
            }
        )
    return pd.DataFrame(rows)
