"""Normalise raw measurements into the [0, 1] components consumed by the
Cooperativity Score (section 22) and build the Allosteric Fingerprint
(section 21).

Normalisation uses a documented logistic squashing function so scores stay
smooth and interpretable rather than hard thresholds. ``scale`` sets the
raw-unit value that maps to ~0.73 (i.e. the "half-saturation-ish" point);
tune per-quantity via config, never hard-code silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from enhancoai.free_energy.cooperativity import CooperativityScoreComponents


def squash(value: float, scale: float) -> float:
    """Monotonic map from [0, inf) raw magnitude to [0, 1)."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    return float(1.0 - np.exp(-abs(value) / scale))


@dataclass
class CooperativityEvidence:
    """Raw, un-normalised measurements that feed the composite score.

    energetic_shift: |delta_delta_g_coop| in kcal/mol.
    dynamic_shift: change in mean RMSF (A) of TF-A between the two systems.
    orientation_shift: change in TF-A/DNA orientation angle (degrees).
    dna_pathway_strength: strongest DNA-mediated communication pathway
        strength in [0, 1] (already normalised by
        :mod:`enhancoai.allostery.pathways`).
    interface_persistence: mean contact-frequency of the TF-A/DNA interface
        residues across frames, already in [0, 1].
    """

    energetic_shift: float
    dynamic_shift: float
    orientation_shift: float
    dna_pathway_strength: float
    interface_persistence: float


def evidence_to_components(
    evidence: CooperativityEvidence,
    energetic_scale: float = 2.0,
    dynamic_scale: float = 1.0,
    orientation_scale: float = 20.0,
) -> CooperativityScoreComponents:
    return CooperativityScoreComponents(
        energetic_coupling=squash(evidence.energetic_shift, energetic_scale),
        dynamic_coupling=squash(evidence.dynamic_shift, dynamic_scale),
        orientation_coupling=squash(evidence.orientation_shift, orientation_scale),
        dna_communication=float(np.clip(evidence.dna_pathway_strength, 0.0, 1.0)),
        interface_persistence=float(np.clip(evidence.interface_persistence, 0.0, 1.0)),
    )


ALLOSTERIC_FINGERPRINT_FIELDS = [
    "global_bend_deg",
    "twist_mean",
    "twist_std",
    "roll_mean",
    "roll_std",
    "minor_groove_width_mean",
    "major_groove_width_mean",
    "orientation_angle_deg",
    "contact_persistence_mean",
    "protein_rmsf_mean",
    "dna_rmsf_mean",
    "mean_abs_correlation",
    "interface_area_a2",
]


def build_allosteric_fingerprint(feature_dict: dict[str, float]) -> pd.Series:
    """Assemble the Allosteric Fingerprint (section 21) from named feature sources.

    Missing fields are recorded as NaN rather than silently dropped so
    fingerprint comparisons across complexes stay well-defined.
    """
    values = {field: feature_dict.get(field, np.nan) for field in ALLOSTERIC_FINGERPRINT_FIELDS}
    return pd.Series(values, name="allosteric_fingerprint")


def compare_fingerprints(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Element-wise, NaN-aware comparison of two Allosteric Fingerprints."""
    return pd.DataFrame({"complex_a": a, "complex_b": b, "delta": b - a})
