import numpy as np

from phase2.interaction_generator import SparseInteractingCTMC
from phase2.observation_operators import ShiftEnvironment, TiltPanelEnvironment, genotype_features
from phase2.jacobian_identifiability import analyze_environments, finite_difference_environment


THETA = np.array([-1.0, -0.7, -0.4, 0.9, -0.65])


def test_independent_limit_matches_product_bernoulli():
    model = SparseInteractingCTMC(3, ())
    baseline = np.array([-1.1, -0.5, 0.2])
    p = model.distribution(baseline)
    event_p = 1.0 - np.exp(-np.exp(baseline))
    expected = np.prod(np.where(model.states == 1, event_p, 1 - event_p), axis=1)
    np.testing.assert_allclose(p, expected, atol=2e-14)


def test_ctmc_analytic_jacobian_matches_finite_difference():
    model = SparseInteractingCTMC(3, ((1, 0), (2, 1)))
    _, analytic = model.distribution_and_jacobian(THETA)
    numeric = model.finite_difference_jacobian(THETA)
    np.testing.assert_allclose(analytic, numeric, atol=1e-9, rtol=1e-7)


def test_tilt_panel_jacobians_match_finite_difference():
    model = SparseInteractingCTMC(3, ((1, 0), (2, 1)))
    features = genotype_features(model.states, [(0,), (0, 1)])
    env = TiltPanelEnvironment(features, np.array([1, 1, 1]))
    phi = np.array([0.2, -0.3])
    _, jt, jp = env.probabilities_and_jacobians(model, THETA, phi)
    nt, np_ = finite_difference_environment(env, model, THETA, phi)
    np.testing.assert_allclose(jt, nt, atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(jp, np_, atol=1e-9, rtol=1e-7)


def test_shift_complementarity_rescues_interaction_direction():
    model = SparseInteractingCTMC(3, ((1, 0), (2, 1)))
    eye = np.eye(model.parameter_count)
    single = ShiftEnvironment(eye[:, 3])
    other = ShiftEnvironment(eye[:, 4])
    single_result = analyze_environments([
        single.probabilities_and_jacobians(model, THETA, np.zeros(1))
    ])
    multi_result = analyze_environments([
        single.probabilities_and_jacobians(model, THETA, np.zeros(1)),
        other.probabilities_and_jacobians(model, THETA, np.zeros(1)),
    ])
    assert single_result.rank == 4
    assert multi_result.rank == 5


def test_redundant_environment_does_not_add_rank():
    model = SparseInteractingCTMC(3, ((1, 0), (2, 1)))
    env = ShiftEnvironment(np.eye(model.parameter_count)[:, 3])
    output = env.probabilities_and_jacobians(model, THETA, np.zeros(1))
    assert analyze_environments([output, output]).rank == 4


def test_realistic_tilts_rescue_weakly():
    model = SparseInteractingCTMC(3, ((1, 0), (2, 1)))
    features = genotype_features(model.states, [(0,), (1,), (2,)])
    e1 = TiltPanelEnvironment(features[:, [0, 1]])
    e2 = TiltPanelEnvironment(features[:, [0, 2]])
    r1 = analyze_environments([e1.probabilities_and_jacobians(model, THETA, np.zeros(2))])
    r2 = analyze_environments([
        e1.probabilities_and_jacobians(model, THETA, np.zeros(2)),
        e2.probabilities_and_jacobians(model, THETA, np.zeros(2)),
    ])
    assert r1.rank == 4
    assert r2.rank == 5
    assert 0 < r2.min_singular_value < 0.01

