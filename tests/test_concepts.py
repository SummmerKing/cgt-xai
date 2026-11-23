"""Solution concepts on small games with known answers.

    python -m pytest tests -q
"""
import itertools

import numpy as np
import pytest

from cgt_xai.concepts.banzhaf import exact_banzhaf_index
from cgt_xai.concepts.gately_point import exact_gately_value
from cgt_xai.concepts.least_core import exact_least_core
from cgt_xai.concepts.nucleolus import exact_nucleolus
from cgt_xai.concepts.shapley import exact_shapley_value, montecarlo_shapley_value


class Players:
    def __init__(self, n):
        self.indices = np.arange(n)
        self.data_names = [f"x{i}" for i in range(n)]

    def __len__(self):
        return len(self.indices)


class Game:
    """Coalition value v(S) as a callable with the interface the solvers expect."""

    def __init__(self, n, v):
        self.data = Players(n)
        self._v = v

    def __call__(self, S):
        return float(self._v(frozenset(S)))


def additive(weights):
    return Game(len(weights), lambda S: sum(weights[i] for i in S))


def glove():
    # player 0 holds a left glove, players 1 and 2 a right glove; a pair is worth 1
    return Game(3, lambda S: float(0 in S and (1 in S or 2 in S)))


def majority():
    return Game(3, lambda S: float(len(S) >= 2))


def test_shapley_additive_game_returns_weights():
    w = [1.0, 2.0, 3.5]
    assert np.allclose(exact_shapley_value(additive(w)).values, w)


def test_shapley_glove_game():
    assert np.allclose(exact_shapley_value(glove()).values, [2 / 3, 1 / 6, 1 / 6])


def test_shapley_efficiency_and_symmetry():
    game = Game(4, lambda S: len(S) ** 2)
    phi = exact_shapley_value(game).values
    assert np.isclose(phi.sum(), 16.0)
    assert np.allclose(phi, phi[0])


def test_montecarlo_shapley_converges():
    np.random.seed(0)
    phi = montecarlo_shapley_value(glove(), n_iterations=4000).values
    assert np.allclose(phi, [2 / 3, 1 / 6, 1 / 6], atol=0.03)


def test_banzhaf_dummy_player_gets_zero():
    game = Game(3, lambda S: float(0 in S and 1 in S))
    beta = exact_banzhaf_index(game, normalized=True).values
    assert np.isclose(beta[2], 0.0)
    assert np.isclose(beta[0], beta[1])


def test_least_core_of_majority_game():
    res = exact_least_core(majority())
    assert np.isclose(res.values.sum(), 1.0, atol=1e-6)
    assert np.allclose(res.values, 1 / 3, atol=1e-4)
    assert res.subsidy == pytest.approx(1 / 3, abs=1e-4)


def test_least_core_is_stable_when_core_is_nonempty():
    res = exact_least_core(glove())
    x = res.values
    for r in range(1, 3):
        for S in itertools.combinations(range(3), r):
            assert x[list(S)].sum() >= glove()(S) - res.subsidy - 1e-6


def test_nucleolus_symmetric_game_splits_equally():
    assert np.allclose(exact_nucleolus(majority()).values, 1 / 3, atol=1e-4)


def test_gately_is_efficient():
    game = Game(3, lambda S: {0: 0, 1: 0, 2: 4, 3: 9}[len(S)])
    x = exact_gately_value(game).values
    assert np.isclose(x.sum(), 9.0)
    assert np.allclose(x, 3.0)
