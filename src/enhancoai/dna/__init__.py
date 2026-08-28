"""DNA structural analysis: base-pair geometry, helical parameters, curvature, grooves.

IMPORTANT: these are simplified, geometry-only calculations intended for
exploratory analysis when a dedicated tool (3DNA, Curves+) is not available.
They are documented approximations, not a reimplementation of the full
Olson/Lu local base-pair-step convention. See ``docs/dna_allostery.md`` for
the exact definitions and their limitations.
"""

from enhancoai.dna.geometry import BasePairFrame, find_watson_crick_pairs, base_pair_frames

__all__ = ["BasePairFrame", "find_watson_crick_pairs", "base_pair_frames"]
