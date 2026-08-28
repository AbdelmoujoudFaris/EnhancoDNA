"""Protein-DNA Cooperativity Interaction Map (section 33).

Combines a residue scan and a DNA scan result into the matrix:

                 DNA position
             1  2  3  4  5  6
Residue A1   .  .  +  -  +  .
Residue A2   .  +  +  -  .  .

`+` = mutation increases predicted cooperativity beyond `threshold`,
`-` = decreases beyond `threshold`, `.` = no strong effect.
"""

from __future__ import annotations

import pandas as pd


def build_interaction_map(
    residue_deltas: pd.DataFrame,
    dna_deltas: pd.DataFrame,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Outer-product style matrix of combined single-mutation effects.

    This is a simplification: it reports the *sum* of the two single-site
    effects rather than a jointly re-evaluated double mutant, since the
    double-mutant space grows combinatorially. It is intended to flag
    candidate residue/base pairs worth a joint re-evaluation, not to
    substitute for one.
    """
    if residue_deltas.empty or dna_deltas.empty:
        return pd.DataFrame()

    residue_labels = residue_deltas["mutation"].tolist()
    dna_labels = dna_deltas["mutation"].tolist()
    matrix = pd.DataFrame(index=residue_labels, columns=dna_labels, dtype=object)

    for _, r_row in residue_deltas.iterrows():
        for _, d_row in dna_deltas.iterrows():
            combined = (r_row["delta"] or 0.0) + (d_row["delta"] or 0.0)
            if combined > threshold:
                symbol = "+"
            elif combined < -threshold:
                symbol = "-"
            else:
                symbol = "."
            matrix.loc[r_row["mutation"], d_row["mutation"]] = symbol

    return matrix
