"""Tests for simulation/market_simulator.py."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.errors import SPTInvariantError
from quantspt.models.gbm import CorrelatedGBMMarket
from quantspt.simulation.market_simulator import MarketSimulation, simulate_market

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gbm_market():
    """3-asset GBM market model."""
    mu = np.array([0.05, 0.07, 0.06])
    cov = np.array(
        [
            [0.04, 0.005, 0.003],
            [0.005, 0.09, 0.01],
            [0.003, 0.01, 0.0625],
        ]
    )
    return CorrelatedGBMMarket(mu=mu, cov=cov)


@pytest.fixture()
def initial_prices():
    return np.array([100.0, 150.0, 80.0])


# ---------------------------------------------------------------------------
# Tests: Basic simulation
# ---------------------------------------------------------------------------


class TestSimulateMarket:
    def test_result_type(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        result = simulate_market(
            gbm_market, x0=initial_prices, T=1.0, n_steps=50, seed=42
        )
        assert isinstance(result.data, MarketSimulation)
        assert result.computation_time_ms > 0

    def test_shapes(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        result = simulate_market(
            gbm_market, x0=initial_prices, T=1.0, n_steps=100, seed=42
        )
        sim = result.data
        assert sim.times.shape == (101,)
        assert sim.prices.shape == (101, 3)
        assert sim.weights.shape == (101, 3)
        assert sim.ranks.shape == (101, 3)

    def test_initial_values(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        result = simulate_market(
            gbm_market, x0=initial_prices, T=1.0, n_steps=50, seed=42
        )
        assert_allclose(result.data.prices[0], initial_prices)

    def test_weights_sum_to_one(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        result = simulate_market(
            gbm_market, x0=initial_prices, T=1.0, n_steps=100, seed=42
        )
        row_sums = result.data.weights.sum(axis=1)
        assert_allclose(row_sums, 1.0, atol=1e-14)

    def test_weights_non_negative(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        result = simulate_market(
            gbm_market, x0=initial_prices, T=1.0, n_steps=100, seed=42
        )
        assert np.all(result.data.weights >= 0)

    def test_prices_positive(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        result = simulate_market(
            gbm_market, x0=initial_prices, T=2.0, n_steps=200, seed=42
        )
        assert np.all(result.data.prices > 0)

    def test_ranks_valid(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        """Each time step should have ranks as a permutation of {0, 1, 2}."""
        result = simulate_market(
            gbm_market, x0=initial_prices, T=1.0, n_steps=50, seed=42
        )
        for t in range(result.data.ranks.shape[0]):
            assert set(result.data.ranks[t]) == {0, 1, 2}

    def test_ranks_consistent_with_weights(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        """Rank 0 should correspond to the largest weight."""
        result = simulate_market(
            gbm_market, x0=initial_prices, T=1.0, n_steps=50, seed=42
        )
        sim = result.data
        for t in range(sim.ranks.shape[0]):
            largest_idx = np.argmax(sim.weights[t])
            assert sim.ranks[t, largest_idx] == 0


# ---------------------------------------------------------------------------
# Tests: Local time tracking
# ---------------------------------------------------------------------------


class TestLocalTimes:
    def test_local_times_shape(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        result = simulate_market(
            gbm_market,
            x0=initial_prices,
            T=1.0,
            n_steps=100,
            seed=42,
            track_local_times=True,
        )
        lt = result.data.local_times
        assert lt is not None
        assert lt.shape == (101, 2)

    def test_local_times_non_negative(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        result = simulate_market(
            gbm_market,
            x0=initial_prices,
            T=1.0,
            n_steps=100,
            seed=42,
            track_local_times=True,
        )
        assert np.all(result.data.local_times >= 0)

    def test_local_times_monotone(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        """Cumulative local times should be non-decreasing."""
        result = simulate_market(
            gbm_market,
            x0=initial_prices,
            T=1.0,
            n_steps=200,
            seed=42,
            track_local_times=True,
        )
        lt = result.data.local_times
        diffs = np.diff(lt, axis=0)
        assert np.all(diffs >= -1e-14)

    def test_no_local_times_by_default(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        result = simulate_market(
            gbm_market,
            x0=initial_prices,
            T=1.0,
            n_steps=50,
            seed=42,
        )
        assert result.data.local_times is None


# ---------------------------------------------------------------------------
# Tests: Metadata and reproducibility
# ---------------------------------------------------------------------------


class TestMarketSimMetadata:
    def test_metadata_keys(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        result = simulate_market(
            gbm_market, x0=initial_prices, T=1.0, n_steps=50, seed=42
        )
        assert result.metadata["model"] == "CorrelatedGBMMarket"
        assert result.metadata["n_assets"] == 3
        assert result.metadata["T"] == 1.0

    def test_reproducibility(
        self,
        gbm_market: CorrelatedGBMMarket,
        initial_prices: np.ndarray,
    ) -> None:
        r1 = simulate_market(gbm_market, x0=initial_prices, T=1.0, n_steps=50, seed=42)
        r2 = simulate_market(gbm_market, x0=initial_prices, T=1.0, n_steps=50, seed=42)
        assert_allclose(r1.data.prices, r2.data.prices)

    def test_invalid_x0(self, gbm_market: CorrelatedGBMMarket) -> None:
        with pytest.raises(SPTInvariantError):
            simulate_market(
                gbm_market,
                x0=np.array([-1.0, 100.0, 50.0]),
                T=1.0,
                n_steps=50,
            )
