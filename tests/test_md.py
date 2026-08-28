from enhancoai.md.loader import load_trajectory
from enhancoai.md.rmsd import compute_rmsd
from enhancoai.md.rmsf import compute_rmsf
from enhancoai.md.contacts import center_of_mass_distance
from enhancoai.md.correlations import dynamic_cross_correlation


def test_load_trajectory_single_frame(demo_two_factor_pdb):
    handle = load_trajectory(demo_two_factor_pdb)
    assert handle.n_frames == 1


def test_rmsd_single_frame_is_zero(demo_two_factor_pdb):
    handle = load_trajectory(demo_two_factor_pdb)
    rmsd = compute_rmsd(handle, selection="protein and name CA")
    assert len(rmsd) == 1
    assert abs(rmsd["rmsd"].iloc[0]) < 1e-6


def test_rmsf_single_frame_is_zero(demo_two_factor_pdb):
    handle = load_trajectory(demo_two_factor_pdb)
    rmsf = compute_rmsf(handle, selection="protein and name CA")
    assert (rmsf["rmsf"].to_numpy() < 1e-6).all()


def test_com_distance(demo_two_factor_pdb):
    handle = load_trajectory(demo_two_factor_pdb)
    com = center_of_mass_distance(handle, "protein and name CA", "nucleic")
    assert len(com) == 1
    assert com["com_distance"].iloc[0] > 0


def test_dccm_shape(demo_two_factor_pdb):
    handle = load_trajectory(demo_two_factor_pdb)
    dccm = dynamic_cross_correlation(handle, selection="protein and name CA")
    n_ca = len(handle.universe.select_atoms("protein and name CA"))
    assert dccm.shape == (n_ca, n_ca)
