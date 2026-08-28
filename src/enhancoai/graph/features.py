"""Per-node feature vectors: protein residues and DNA nucleotides.

Node feature layout (14-dim, fixed so the GNN's input layer size is stable):
    [0:20]  -- not used directly; see RESIDUE_VOCAB one-hot below (20 protein + 4 dna = 24 total node type slots)
Actual layout (see `node_feature_vector`):
    0-23   one-hot residue/nucleotide identity (20 amino acids + 4 DNA bases)
    24     is_protein flag
    25     is_dna flag
    26     hydrophobicity (protein) or 0
    27     charge proxy
"""

from __future__ import annotations

import numpy as np

from enhancoai.voxel.representation import HYDROPHOBICITY

AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
DNA_BASES = ["DA", "DT", "DG", "DC"]
RESIDUE_VOCAB = AMINO_ACIDS + DNA_BASES

CHARGE = {
    "ARG": 1.0, "LYS": 1.0, "HIS": 0.5,
    "ASP": -1.0, "GLU": -1.0,
    "DA": 0.0, "DT": 0.0, "DG": 0.0, "DC": 0.0,
}

NODE_FEATURE_DIM = len(RESIDUE_VOCAB) + 4  # one-hot identity + is_protein + is_dna + hydrophobicity + charge


def node_feature_vector(res_name: str, is_protein: bool) -> np.ndarray:
    res_name = res_name.strip().upper()
    one_hot = np.zeros(len(RESIDUE_VOCAB), dtype=np.float32)
    if res_name in RESIDUE_VOCAB:
        one_hot[RESIDUE_VOCAB.index(res_name)] = 1.0

    extra = np.array(
        [
            1.0 if is_protein else 0.0,
            0.0 if is_protein else 1.0,
            HYDROPHOBICITY.get(res_name, 0.0) if is_protein else 0.0,
            CHARGE.get(res_name, 0.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([one_hot, extra])
