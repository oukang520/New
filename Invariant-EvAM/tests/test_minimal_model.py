import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.special import expit
from minimal_model import (
    arbitrary_weight_alternative, bounded_profile_intervals, build_design,
    identifiable_theta_dimension, product_distribution, rates_to_theta,
    states, theta_to_rates, tilted_distribution, weighted_distribution,
)


def test_ctmc_snapshot_mapping_roundtrip():
    theta = np.array([-1.2, -0.3, 0.7, 1.1])
    rates = theta_to_rates(theta)
    assert np.all(rates > 0)
    assert np.allclose(rates_to_theta(rates), theta)
    assert np.allclose(1 - np.exp(-rates), expit(theta))


def test_exponential_tilt_matches_shifted_logit():
    theta = np.array([-1.1, 0.2, 0.8])
    x = states(3)
    base = product_distribution(theta)
    for j in range(3):
        got = weighted_distribution(base, x[:, j], 1.3)
        expected = tilted_distribution(theta, j, 1.3)
        assert np.allclose(got, expected, atol=1e-13)


def test_complementary_and_redundant_rank():
    full2 = np.ones((2, 4), dtype=bool)
    assert identifiable_theta_dimension(build_design([0], np.ones((1, 4), bool))) == 3
    assert identifiable_theta_dimension(build_design([0, 1], full2)) == 4
    assert identifiable_theta_dimension(build_design([0, 0], full2)) == 3
    failed_panel = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], bool)
    assert identifiable_theta_dimension(build_design([0, 1], failed_panel)) == 2


def test_population_profile_width_shrinks_only_for_complementary_design():
    theta = np.array([-1.35, -0.65, 0.15, 0.8])
    for biases, expected_first_width in [([0], 4.0), ([0, 0], 4.0), ([0, 1], 0.0)]:
        design = build_design(biases, np.ones((len(biases), 4), bool))
        gamma = np.full(len(biases), 1.2)
        y = np.asarray([theta[j] + (gamma[e] if j == biases[e] else 0.0) for e, j in design.rows])
        lo, hi, ok = bounded_profile_intervals(y, np.zeros(len(y)), design, 2.0, 4.0, population=True)
        assert ok
        assert np.isclose(hi[0] - lo[0], expected_first_width, atol=5e-8)


def test_arbitrary_ascertainment_preserves_every_environment():
    truth = np.array([-1.0, -0.2, 0.4])
    alternative = np.array([0.8, -1.1, 1.3])
    for j, gamma in [(0, 1.2), (1, -0.7), (2, 0.5)]:
        observed = tilted_distribution(truth, j, gamma)
        weights = arbitrary_weight_alternative(observed, alternative)
        reconstructed = product_distribution(alternative) * weights
        reconstructed /= reconstructed.sum()
        assert np.allclose(reconstructed, observed, atol=1e-13)


def test_biology_shift_on_bias_coordinate_is_exactly_confoundable():
    theta = np.array([-1.0, 0.2, 0.9])
    delta = 0.4
    # A biological shift and an ascertainment shift add on exactly the same logit.
    assert np.allclose(
        tilted_distribution(theta, 0, 1.1, np.array([delta, 0, 0])),
        tilted_distribution(theta, 0, 1.1 + delta),
    )
