"""Minor/major groove width estimation from phosphate-phosphate distances.

Uses the standard simplified convention (e.g. El Hassan & Calladine 1998):
raw P-P distance across the groove minus the sum of phosphate van der Waals
diameters (~5.8 A for the minor groove, ~4.7 A for the wider major groove
spacing convention) approximates the true groove width. This is a
geometric approximation intended for comparative/trend analysis, not a
substitute for dedicated groove-geometry software.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MINOR_GROOVE_CORRECTION = 5.8
MAJOR_GROOVE_CORRECTION = 4.7
MINOR_GROOVE_OFFSET = 3  # base pairs offset (i, i+3) approximates minor groove P-P
MAJOR_GROOVE_OFFSET = 7  # base pairs offset (i, i+7) approximates major groove P-P


def _phosphate_positions(structure, chain_id: str, res_seqs: list[int]) -> dict[int, np.ndarray]:
    chain_atoms = structure.chain(chain_id)
    positions = {}
    for res_seq in res_seqs:
        match = chain_atoms[
            (chain_atoms["res_seq"] == res_seq) & (chain_atoms["atom_name"].str.strip() == "P")
        ]
        if not match.empty:
            positions[res_seq] = match[["x", "y", "z"]].to_numpy(dtype=float)[0]
    return positions


def groove_widths(structure, chain_a: str, chain_b: str, pairs: list[tuple[int, int]]) -> pd.DataFrame:
    """Estimate minor/major groove width at each base-pair-step position.

    ``pairs`` should be the Watson-Crick pairs from
    :func:`enhancoai.dna.geometry.find_watson_crick_pairs`, ordered along
    ``chain_a``.
    """
    res_seqs_a = [p[0] for p in pairs]
    res_seqs_b = [p[1] for p in pairs]
    phos_a = _phosphate_positions(structure, chain_a, res_seqs_a)
    phos_b = _phosphate_positions(structure, chain_b, res_seqs_b)

    n = len(pairs)
    rows = []
    for i in range(n):
        res_a_i = res_seqs_a[i]
        minor_j = i + MINOR_GROOVE_OFFSET
        major_j = i + MAJOR_GROOVE_OFFSET

        minor_width = np.nan
        if minor_j < n and res_a_i in phos_a:
            res_b_j = res_seqs_b[minor_j]
            if res_b_j in phos_b:
                minor_width = float(np.linalg.norm(phos_a[res_a_i] - phos_b[res_b_j]) - MINOR_GROOVE_CORRECTION)

        major_width = np.nan
        if major_j < n and res_a_i in phos_a:
            res_b_j = res_seqs_b[major_j]
            if res_b_j in phos_b:
                major_width = float(np.linalg.norm(phos_a[res_a_i] - phos_b[res_b_j]) - MAJOR_GROOVE_CORRECTION)

        rows.append(
            {
                "index": i,
                "res_seq_a": res_a_i,
                "minor_groove_width": minor_width,
                "major_groove_width": major_width,
            }
        )
    return pd.DataFrame(rows)
