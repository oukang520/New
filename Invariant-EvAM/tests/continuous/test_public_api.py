import numpy as np

from invariant_evam import diagnose_identifiability, operator_audit
from invariant_evam.functionals import functional_identifiability_map, recommend_environment
from invariant_evam.robustness import effective_error_interval, linear_functional_interval
from phase2.interaction_generator import SparseInteractingCTMC
from phase2.observation_operators import ShiftEnvironment


def test_public_diagnostic_and_functional_map():
    model = SparseInteractingCTMC(3, ((1, 0), (2, 1)))
    theta = np.array([-1.0, -.7, -.4, .9, -.65])
    eye = np.eye(model.parameter_count)
    envs = [ShiftEnvironment(eye[:, 3]), ShiftEnvironment(eye[:, 4])]
    result = diagnose_identifiability(model, envs, theta)
    assert result.rank == model.parameter_count
    rows = functional_identifiability_map(model, result)
    assert rows and all(row["identified"] for row in rows)
    assert all(row["finite_operator"] for row in operator_audit(envs, model))


def test_environment_recommendation_prefers_rank_rescue():
    model = SparseInteractingCTMC(3, ((1, 0), (2, 1)))
    theta = np.array([-1.0, -.7, -.4, .9, -.65]); eye = np.eye(5)
    current = ShiftEnvironment(eye[:, 3]).probabilities_and_jacobians(model, theta, np.zeros(1))
    candidates = {
        "duplicate": ShiftEnvironment(eye[:, 3]).probabilities_and_jacobians(model, theta, np.zeros(1)),
        "complement": ShiftEnvironment(eye[:, 4]).probabilities_and_jacobians(model, theta, np.zeros(1)),
    }
    ranked = recommend_environment([current], candidates)
    assert ranked[0]["candidate"] == "complement"
    assert ranked[0]["rank"] == 5


def test_robustness_intervals_obey_contract():
    assert np.allclose(effective_error_interval(.9, .2, .1), (.6, 1.2))
    low, high = linear_functional_interval(np.array([1., -1.]), np.array([.7, .2]), .1, .05)
    assert np.isclose(low, .2) and np.isclose(high, .8)
