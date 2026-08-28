import numpy as np
import pytest

from enhancoai.free_energy.pmf import compute_pmf_1d, compute_pmf_2d, free_energy_at_extremes
from enhancoai.free_energy.cooperativity import (
    binding_free_energy,
    cooperativity_free_energy,
    CooperativityScoreComponents,
    CooperativityScoreWeights,
    compute_cooperativity_score,
)


def test_pmf_1d_basic_shape():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, 1000)
    pmf = compute_pmf_1d(samples, n_bins=20)
    assert len(pmf.bin_centers) == 20
    assert np.nanmin(pmf.free_energy) >= -1e-9  # shifted so the minimum is ~0


def test_pmf_1d_small_sample_warns():
    pmf = compute_pmf_1d(np.array([1.0, 2.0, 3.0]), n_bins=5)
    assert any("unconverged" in w for w in pmf.warnings)


def test_pmf_2d_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    y = rng.normal(0, 1, 500)
    x_edges, y_edges, free_energy = compute_pmf_2d(x, y, n_bins=10)
    assert free_energy.shape == (10, 10)


def test_binding_free_energy_distinguishes_bound_unbound():
    rng = np.random.default_rng(1)
    # a coordinate concentrated near 0 (bound) with a long unbound tail
    samples = np.concatenate([rng.normal(1, 0.5, 900), rng.uniform(20, 25, 100)])
    pmf = compute_pmf_1d(samples, n_bins=30)
    g_bind = binding_free_energy(pmf, bound_state="low")
    assert g_bind < 0  # bound end should be more probable/favourable than the tail


def test_cooperativity_free_energy_not_trivially_zero():
    rng = np.random.default_rng(2)
    samples_a = np.concatenate([rng.normal(15, 2, 900), rng.uniform(0, 2, 100)])
    samples_ab = np.concatenate([rng.normal(2, 1, 900), rng.uniform(15, 20, 100)])
    pmf_a = compute_pmf_1d(samples_a, n_bins=30)
    pmf_ab = compute_pmf_1d(samples_ab, n_bins=30)
    result = cooperativity_free_energy(pmf_a, pmf_ab, bound_state="low")
    assert result.delta_delta_g_coop != 0.0


def test_cooperativity_score_weights_normalise():
    weights = CooperativityScoreWeights(1, 1, 1, 1, 1).normalised()
    total = (
        weights.energetic_coupling + weights.dynamic_coupling + weights.orientation_coupling
        + weights.dna_communication + weights.interface_persistence
    )
    assert abs(total - 1.0) < 1e-9


def test_cooperativity_score_rejects_out_of_range_components():
    components = CooperativityScoreComponents(1.5, 0.5, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        compute_cooperativity_score(components)


def test_cooperativity_score_is_weighted_average():
    components = CooperativityScoreComponents(1.0, 0.0, 0.0, 0.0, 0.0)
    weights = CooperativityScoreWeights(1.0, 0.0, 0.0, 0.0, 0.0)
    result = compute_cooperativity_score(components, weights)
    assert abs(result.total_score - 1.0) < 1e-9
