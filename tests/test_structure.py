from enhancoai.structure.chain_detection import ChainType, classify_chains, classify_chain
from enhancoai.structure.cleaning import report_missing_backbone_atoms


def test_load_structure_atom_count(demo_structure):
    assert demo_structure.n_atoms() > 0
    assert demo_structure.n_models() == 1


def test_chain_ids_present(demo_structure):
    assert set(demo_structure.chain_ids) == {"A", "B", "C", "D"}


def test_classify_chains(demo_structure):
    classification = classify_chains(demo_structure)
    assert classification["A"] == ChainType.PROTEIN
    assert classification["B"] == ChainType.PROTEIN
    assert classification["C"] == ChainType.DNA
    assert classification["D"] == ChainType.DNA


def test_classify_chain_empty():
    assert classify_chain([]) == ChainType.WATER


def test_classify_chain_protein():
    assert classify_chain(["ALA", "GLY", "SER"] * 5) == ChainType.PROTEIN


def test_classify_chain_dna():
    assert classify_chain(["DA", "DT", "DG", "DC"] * 3) == ChainType.DNA


def test_missing_backbone_report_is_list(demo_structure):
    report = report_missing_backbone_atoms(demo_structure)
    assert isinstance(report, list)


def test_single_factor_has_no_chain_b(demo_single_factor_pdb):
    from enhancoai.structure.parser import load_structure

    structure = load_structure(demo_single_factor_pdb)
    assert "B" not in structure.chain_ids
    assert "A" in structure.chain_ids
