import numpy as np

from enhancoai.dna.geometry import find_watson_crick_pairs, base_pair_frames
from enhancoai.dna.helical_parameters import step_parameters
from enhancoai.dna.curvature import local_curvature, global_bend_angle, end_to_end_contour_ratio
from enhancoai.dna.groove_analysis import groove_widths


def test_find_watson_crick_pairs(demo_structure):
    pairs = find_watson_crick_pairs(demo_structure, "C", "D")
    assert len(pairs) > 0
    for res_a, res_b in pairs:
        assert isinstance(res_a, int) and isinstance(res_b, int)


def test_base_pair_frames_orthonormal(demo_structure):
    pairs = find_watson_crick_pairs(demo_structure, "C", "D")
    frames = base_pair_frames(demo_structure, "C", "D", pairs)
    assert len(frames) == len(pairs)
    for frame in frames:
        assert np.isclose(np.linalg.norm(frame.x_axis), 1.0, atol=1e-3)
        assert np.isclose(np.linalg.norm(frame.z_axis), 1.0, atol=1e-3)
        assert abs(np.dot(frame.x_axis, frame.z_axis)) < 1e-2


def test_step_parameters(demo_structure):
    pairs = find_watson_crick_pairs(demo_structure, "C", "D")
    frames = base_pair_frames(demo_structure, "C", "D", pairs)
    steps = step_parameters(frames)
    assert len(steps) == len(frames) - 1
    for col in ("shift", "slide", "rise", "tilt", "roll", "twist"):
        assert col in steps.columns


def test_curvature_functions(demo_structure):
    pairs = find_watson_crick_pairs(demo_structure, "C", "D")
    frames = base_pair_frames(demo_structure, "C", "D", pairs)
    curvature = local_curvature(frames)
    assert len(curvature) == len(frames)
    assert 0.0 <= global_bend_angle(frames) <= 180.0
    assert 0.0 <= end_to_end_contour_ratio(frames) <= 1.0 + 1e-6


def test_groove_widths(demo_structure):
    pairs = find_watson_crick_pairs(demo_structure, "C", "D")
    widths = groove_widths(demo_structure, "C", "D", pairs)
    assert "minor_groove_width" in widths.columns
    assert "major_groove_width" in widths.columns
