from enhancoai.interactions.contact_maps import find_atom_contacts, contacts_to_frame, residue_contact_map
from enhancoai.interactions.protein_dna import analyse_protein_dna_interactions
from enhancoai.interactions.protein_protein import analyse_protein_protein_interactions
from enhancoai.structure.selection import heavy_atoms


def test_find_atom_contacts_symmetry(demo_structure):
    protein_atoms = heavy_atoms(demo_structure.chain("A"))
    dna_atoms = heavy_atoms(demo_structure.chain("C"))
    contacts = find_atom_contacts(protein_atoms, dna_atoms, cutoff=6.0)
    assert isinstance(contacts, list)
    frame = contacts_to_frame(contacts)
    assert set(frame.columns) >= {"chain_a", "res_seq_a", "chain_b", "res_seq_b", "distance"}


def test_find_atom_contacts_empty_selection(demo_structure):
    empty = demo_structure.chain("A").iloc[0:0]
    dna_atoms = heavy_atoms(demo_structure.chain("C"))
    assert find_atom_contacts(empty, dna_atoms) == []


def test_residue_contact_map_aggregates(demo_structure):
    protein_atoms = heavy_atoms(demo_structure.chain("A"))
    dna_atoms = heavy_atoms(demo_structure.chain("C"))
    contacts = find_atom_contacts(protein_atoms, dna_atoms, cutoff=6.0)
    res_map = residue_contact_map(contacts)
    if not res_map.empty:
        assert (res_map["n_atom_contacts"] >= 1).all()
        assert res_map["min_distance"].is_monotonic_increasing


def test_analyse_protein_dna_interactions(demo_structure):
    result = analyse_protein_dna_interactions(demo_structure, "A", "C", heavy_atom_cutoff=6.0)
    summary = result.summary()
    assert summary["protein_chain"] == "A"
    assert summary["dna_chain"] == "C"
    assert summary["n_interface_protein_residues"] >= 0


def test_analyse_protein_protein_interactions(demo_structure):
    result = analyse_protein_protein_interactions(demo_structure, "A", "B", heavy_atom_cutoff=6.0, compute_sasa=False)
    assert result.chain_a == "A"
    assert result.chain_b == "B"
    assert result.min_distance >= 0
