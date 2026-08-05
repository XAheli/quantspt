"""Tests for rank/transitions.py -- rank transition matrices and mobility."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.errors import SPTInvariantError
from quantspt.rank.transitions import (
    expected_sojourn_times,
    rank_mobility_index,
    rank_transition_matrix,
)


@pytest.fixture()
def stable_weights() -> np.ndarray:
    """Weight paths where ranks never change (static market)."""
    rng = np.random.default_rng(42)
    base = np.array([0.5, 0.3, 0.2])
    noise = rng.normal(0, 0.001, size=(100, 3))
    weights = base + noise
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights


@pytest.fixture()
def shuffled_weights() -> np.ndarray:
    """Weight paths where ranks change frequently."""
    rng = np.random.default_rng(99)
    T, n = 200, 4
    weights = rng.dirichlet(np.ones(n), size=T)
    return weights


class TestRankTransitionMatrix:
    def test_row_stochastic(self, stable_weights: np.ndarray) -> None:
        """Transition matrix rows must sum to 1."""
        P = rank_transition_matrix(stable_weights, horizon=1)
        assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)

    def test_shape(self, stable_weights: np.ndarray) -> None:
        n = stable_weights.shape[1]
        P = rank_transition_matrix(stable_weights, horizon=1)
        assert P.shape == (n, n)

    def test_stable_market_diagonal_dominant(self, stable_weights: np.ndarray) -> None:
        """In a stable market, the diagonal should dominate."""
        P = rank_transition_matrix(stable_weights, horizon=1)
        for i in range(P.shape[0]):
            assert P[i, i] > 0.8

    def test_shuffled_market_off_diagonal(self, shuffled_weights: np.ndarray) -> None:
        """In a highly mobile market, off-diagonal entries should be significant."""
        P = rank_transition_matrix(shuffled_weights, horizon=1)
        n = P.shape[0]
        off_diag_mass = 1.0 - np.trace(P) / n
        assert off_diag_mass > 0.1

    def test_longer_horizon(self, stable_weights: np.ndarray) -> None:
        """Longer horizons should produce valid transition matrices."""
        P5 = rank_transition_matrix(stable_weights, horizon=5)
        assert_allclose(P5.sum(axis=1), 1.0, atol=1e-12)

    def test_nonnegative(self, shuffled_weights: np.ndarray) -> None:
        P = rank_transition_matrix(shuffled_weights)
        assert np.all(P >= 0)

    def test_horizon_1_is_default(self, stable_weights: np.ndarray) -> None:
        P_default = rank_transition_matrix(stable_weights)
        P_h1 = rank_transition_matrix(stable_weights, horizon=1)
        assert_allclose(P_default, P_h1)

    def test_validation_1d(self) -> None:
        with pytest.raises(SPTInvariantError):
            rank_transition_matrix(np.array([0.5, 0.3, 0.2]))

    def test_validation_horizon_too_large(self, stable_weights: np.ndarray) -> None:
        T = stable_weights.shape[0]
        with pytest.raises(SPTInvariantError):
            rank_transition_matrix(stable_weights, horizon=T)

    def test_validation_horizon_zero(self, stable_weights: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            rank_transition_matrix(stable_weights, horizon=0)

    def test_validation_single_stock(self) -> None:
        with pytest.raises(SPTInvariantError):
            rank_transition_matrix(np.ones((10, 1)))

    def test_two_stocks(self) -> None:
        """Two stocks: ranks are deterministic from weights."""
        weights = np.array(
            [
                [0.6, 0.4],
                [0.55, 0.45],
                [0.45, 0.55],
                [0.4, 0.6],
                [0.6, 0.4],
            ]
        )
        P = rank_transition_matrix(weights, horizon=1)
        assert P.shape == (2, 2)
        assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)


class TestExpectedSojournTimes:
    def test_stable_market_long_sojourn(self, stable_weights: np.ndarray) -> None:
        P = rank_transition_matrix(stable_weights)
        sojourn = expected_sojourn_times(P)
        assert np.all(sojourn > 1.0)

    def test_identity_gives_inf(self) -> None:
        """If P = I, stocks never leave their rank => infinite sojourn."""
        P = np.eye(3)
        sojourn = expected_sojourn_times(P)
        assert np.all(np.isinf(sojourn))

    def test_shape(self) -> None:
        P = np.array([[0.8, 0.2], [0.3, 0.7]])
        sojourn = expected_sojourn_times(P)
        assert sojourn.shape == (2,)

    def test_known_value(self) -> None:
        """P_{kk}=0.8 => sojourn = 1/(1-0.8) = 5."""
        P = np.array([[0.8, 0.2], [0.3, 0.7]])
        sojourn = expected_sojourn_times(P)
        assert_allclose(sojourn[0], 5.0)
        assert_allclose(sojourn[1], 1.0 / 0.3)

    def test_validation_1d(self) -> None:
        with pytest.raises(SPTInvariantError):
            expected_sojourn_times(np.array([0.5, 0.3]))

    def test_validation_non_square(self) -> None:
        with pytest.raises(SPTInvariantError):
            expected_sojourn_times(np.ones((2, 3)))


class TestRankMobilityIndex:
    def test_identity_gives_zero(self) -> None:
        """No rank changes => mobility = 0."""
        P = np.eye(5)
        assert rank_mobility_index(P) == 0.0

    def test_uniform_gives_max(self) -> None:
        """Uniform transition => mobility = 1 - 1/n."""
        n = 4
        P = np.ones((n, n)) / n
        assert_allclose(rank_mobility_index(P), 1.0 - 1.0 / n)

    def test_range(self, shuffled_weights: np.ndarray) -> None:
        P = rank_transition_matrix(shuffled_weights)
        M = rank_mobility_index(P)
        assert 0.0 <= M <= 1.0 + 1e-10

    def test_validation(self) -> None:
        with pytest.raises(SPTInvariantError):
            rank_mobility_index(np.array([1.0]))

    def test_consistent_with_sojourn(self) -> None:
        """Higher mobility => lower average sojourn times."""
        P_low = np.array([[0.95, 0.05], [0.05, 0.95]])
        P_high = np.array([[0.5, 0.5], [0.5, 0.5]])
        m_low = rank_mobility_index(P_low)
        m_high = rank_mobility_index(P_high)
        s_low = np.mean(expected_sojourn_times(P_low))
        s_high = np.mean(expected_sojourn_times(P_high))
        assert m_low < m_high
        assert s_low > s_high
