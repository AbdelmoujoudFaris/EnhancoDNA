"""Voxel channel definitions for the 3D CNN interface representation (section 24)."""

from __future__ import annotations

CHANNEL_NAMES = [
    "protein_occupancy",
    "dna_occupancy",
    "electrostatic_potential",
    "hydrophobicity",
    "hydrogen_bond_donor",
    "hydrogen_bond_acceptor",
    "dna_backbone",
    "base_identity",
    "interface_density",
    "distance_field",
]

N_CHANNELS = len(CHANNEL_NAMES)

# Simplified per-atom-name partial charge proxy (unitless, sign-only meaningful).
PARTIAL_CHARGE = {
    "NZ": 1.0, "NH1": 1.0, "NH2": 1.0, "NE": 0.5, "ND1": 0.5, "NE2": 0.5,
    "OD1": -1.0, "OD2": -1.0, "OE1": -1.0, "OE2": -1.0,
    "OP1": -1.0, "OP2": -1.0, "O1P": -1.0, "O2P": -1.0, "P": 1.0,
}

# Kyte-Doolittle hydrophobicity scale.
HYDROPHOBICITY = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9,
    "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9, "TYR": -1.3,
    "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5, "ASP": -3.5, "ASN": -3.5,
    "LYS": -3.9, "ARG": -4.5,
}

# H-bond donor/acceptor atom-name sets (heavy-atom proxy, see enhancoai.interactions.hydrogen_bonds).
DONOR_ATOMS = {"N", "NZ", "NH1", "NH2", "NE", "NE1", "NE2", "ND1", "ND2", "OG", "OG1", "OH", "SG"}
ACCEPTOR_ATOMS = {"O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "SD", "OP1", "OP2", "O1P", "O2P"}

DNA_BACKBONE_ATOMS = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}
