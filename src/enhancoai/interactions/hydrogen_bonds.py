"""Hydrogen-bond and salt-bridge detection between arbitrary atom selections.

Hydrogen positions are frequently absent from crystallographic protein-DNA
structures, so hydrogen bonds are identified with a donor/acceptor
heavy-atom distance criterion (Baker & Hubbard-style heavy-atom distance,
default <= 3.5 A) rather than requiring an explicit D-H...A angle. This is
a standard, documented simplification -- not a claim of full electronic
accuracy -- and is noted wherever results are reported.
"""

from __future__ import annotations

import pandas as pd

from enhancoai.interactions.contact_maps import find_atom_contacts, contacts_to_frame

# Polar atom names commonly acting as H-bond donors/acceptors (protein + DNA).
DONOR_ACCEPTOR_ELEMENTS = {"N", "O", "S"}

POSITIVE_ATOMS = {
    ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE"),
    ("LYS", "NZ"),
    ("HIS", "ND1"), ("HIS", "NE2"),
}
NEGATIVE_ATOMS = {
    ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"),
}
DNA_NEGATIVE_ATOMS = {"OP1", "OP2", "O1P", "O2P"}


def find_hydrogen_bonds(atoms_a: pd.DataFrame, atoms_b: pd.DataFrame, distance_cutoff: float = 3.5) -> pd.DataFrame:
    """Heavy-atom-distance proxy for hydrogen bonds between polar atoms."""
    polar_a = atoms_a[atoms_a["element"].str.upper().isin(DONOR_ACCEPTOR_ELEMENTS)]
    polar_b = atoms_b[atoms_b["element"].str.upper().isin(DONOR_ACCEPTOR_ELEMENTS)]
    contacts = find_atom_contacts(polar_a, polar_b, cutoff=distance_cutoff)
    frame = contacts_to_frame(contacts)
    if not frame.empty:
        frame = frame.assign(interaction_type="hydrogen_bond (heavy-atom distance proxy)")
    return frame


def find_salt_bridges(atoms_a: pd.DataFrame, atoms_b: pd.DataFrame, distance_cutoff: float = 4.0) -> pd.DataFrame:
    """Detect charged-atom pairs (protein Arg/Lys/His <-> Asp/Glu or DNA phosphate oxygens)."""

    def _charged(atoms: pd.DataFrame, charge_set: set, extra_names: set | None = None) -> pd.DataFrame:
        key = list(zip(atoms["res_name"].str.upper(), atoms["atom_name"].str.strip()))
        mask = [k in charge_set for k in key]
        if extra_names:
            mask = [m or (name.strip() in extra_names) for m, name in zip(mask, atoms["atom_name"])]
        return atoms[mask]

    pos_a = _charged(atoms_a, POSITIVE_ATOMS)
    neg_b = _charged(atoms_b, NEGATIVE_ATOMS, DNA_NEGATIVE_ATOMS)
    pos_b = _charged(atoms_b, POSITIVE_ATOMS)
    neg_a = _charged(atoms_a, NEGATIVE_ATOMS, DNA_NEGATIVE_ATOMS)

    contacts = find_atom_contacts(pos_a, neg_b, cutoff=distance_cutoff) + find_atom_contacts(
        pos_b, neg_a, cutoff=distance_cutoff
    )
    frame = contacts_to_frame(contacts)
    if not frame.empty:
        frame = frame.assign(interaction_type="salt_bridge")
    return frame
