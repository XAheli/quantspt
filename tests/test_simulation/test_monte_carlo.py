"""Tests for simulation/monte_carlo.py — Monte Carlo engine."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.core.processes import CorrelatedGBM
from quantspt.errors import SPTInvariantError
from quantspt.simulation.monte_carlo import MonteCarloEngine, MonteCarloResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scalar_gbm():
    """1-D GBM: μ=0.05, σ=0.2, S(0)=100."""
    return CorrelatedGBM(
        mu=np.array([0.05]),
        cov=np.array([[0.04]]),
        x0=np.array([100.0]),
    )


@pytest.fixture()
def two_asset_gbm():
    """2-D correlated GBM."""
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    return CorrelatedGBM(
        mu=np.array([0.05, 0.08]),
        cov=cov,
        x0=np.array([100.0, 100.0]),
    )


# ---------------------------------------------------------------------------
# Tests: Basic MC
# ---------------------------------------------------------------------------


class TestMonteCarloBasic:
    def test_result_type(self, scalar_gbm: CorrelatedGBM) -> None:
        engine = MonteCarloEngine(scalar_gbm, n_paths=50, seed=42)
        result = engine.run()
        assert result.data is not None
        assert isinstance(result.data, MonteCarloResult)
        assert result.computation_time_ms > 0

    def test_terminal_shape(self, scalar_gbm: CorrelatedGBM) -> None:
        engine = MonteCarloEngine(scalar_gbm, n_paths=100, seed=42)
        result = engine.run()
        assert result.data.terminal_values.shape == (100, 1)
        assert result.data.mean.shape == (1,)
        assert result.data.std.shape == (1,)

    def test_multi_asset_shape(self, two_asset_gbm: CorrelatedGBM) -> None:
        engine = MonteCarloEngine(two_asset_gbm, n_paths=50, seed=42)
        result = engine.run()
        assert result.data.terminal_values.shape == (50, 2)
        assert result.data.mean.shape == (2,)

    def test_mean_converges_to_analytical(self, scalar_gbm: CorrelatedGBM) -> None:
        """E[S(T)] = S(0) exp(μT) for GBM."""
        engine = MonteCarloEngine(scalar_gbm, n_paths=5000, T=1.0, n_steps=1, seed=123)
        result = engine.run()
        expected = 100.0 * np.exp(0.05 * 1.0)
        assert_allclose(result.data.mean[0], expected, rtol=0.05)

    def test_confidence_interval_covers_mean(self, scalar_gbm: CorrelatedGBM) -> None:
        """95% CI should cover the analytical mean."""
        engine = MonteCarloEngine(scalar_gbm, n_paths=5000, T=1.0, n_steps=1, seed=99)
        result = engine.run()
        expected = 100.0 * np.exp(0.05)
        assert result.data.ci_lower[0] < expected < result.data.ci_upper[0]

    def test_all_terminals_positive(self, scalar_gbm: CorrelatedGBM) -> None:
        """GBM terminal values are positive."""
        engine = MonteCarloEngine(scalar_gbm, n_paths=200, seed=42)
        result = engine.run()
        assert np.all(result.data.terminal_values > 0)

    def test_metadata(self, scalar_gbm: CorrelatedGBM) -> None:
        engine = MonteCarloEngine(scalar_gbm, n_paths=50, T=2.0, n_steps=100, seed=42)
        result = engine.run()
        assert result.metadata["T"] == 2.0
        assert result.metadata["n_steps"] == 100
        assert result.metadata["antithetic"] is False

    def test_reproducibility(self, scalar_gbm: CorrelatedGBM) -> None:
        """Same seed produces same results."""
        engine1 = MonteCarloEngine(scalar_gbm, n_paths=50, seed=42)
        engine2 = MonteCarloEngine(scalar_gbm, n_paths=50, seed=42)
        r1 = engine1.run()
        r2 = engine2.run()
        assert_allclose(r1.data.mean, r2.data.mean)


# ---------------------------------------------------------------------------
# Tests: Antithetic variates
# ---------------------------------------------------------------------------


class TestAntitheticVariates:
    def test_antithetic_runs(self, scalar_gbm: CorrelatedGBM) -> None:
        engine = MonteCarloEngine(scalar_gbm, n_paths=100, antithetic=True, seed=42)
        result = engine.run()
        assert result.data.antithetic is True
        assert result.data.terminal_values.shape == (100, 1)

    def test_antithetic_mean_converges(self, scalar_gbm: CorrelatedGBM) -> None:
        """Antithetic MC mean should converge to E[S(T)]."""
        engine = MonteCarloEngine(
            scalar_gbm,
            n_paths=5000,
            T=1.0,
            n_steps=1,
            antithetic=True,
            seed=42,
        )
        result = engine.run()
        expected = 100.0 * np.exp(0.05)
        assert_allclose(result.data.mean[0], expected, rtol=0.05)

    def test_variance_reduction(self, scalar_gbm: CorrelatedGBM) -> None:
        """Antithetic variates should produce tighter CIs than plain MC."""
        n_paths = 2000
        engine_plain = MonteCarloEngine(
            scalar_gbm,
            n_paths=n_paths,
            T=1.0,
            n_steps=50,
            seed=42,
        )
        engine_anti = MonteCarloEngine(
            scalar_gbm,
            n_paths=n_paths,
            T=1.0,
            n_steps=50,
            antithetic=True,
            seed=42,
        )

        plain = engine_plain.run()
        anti = engine_anti.run()

        ci_width_plain = float(plain.data.ci_upper[0] - plain.data.ci_lower[0])
        ci_width_anti = float(anti.data.ci_upper[0] - anti.data.ci_lower[0])

        assert ci_width_anti < ci_width_plain, (
            f"Antithetic CI ({ci_width_anti:.4f}) should be tighter than "
            f"plain ({ci_width_plain:.4f})"
        )

    def test_antithetic_multi_asset(self, two_asset_gbm: CorrelatedGBM) -> None:
        engine = MonteCarloEngine(two_asset_gbm, n_paths=100, antithetic=True, seed=42)
        result = engine.run()
        assert result.data.terminal_values.shape == (100, 2)
        assert np.all(np.isfinite(result.data.mean))


# ---------------------------------------------------------------------------
# Tests: Validation
# ---------------------------------------------------------------------------


class TestMonteCarloValidation:
    def test_invalid_n_paths(self, scalar_gbm: CorrelatedGBM) -> None:
        with pytest.raises(SPTInvariantError):
            MonteCarloEngine(scalar_gbm, n_paths=0)

    def test_invalid_T(self, scalar_gbm: CorrelatedGBM) -> None:
        with pytest.raises(SPTInvariantError):
            MonteCarloEngine(scalar_gbm, T=-1.0)

    def test_invalid_n_steps(self, scalar_gbm: CorrelatedGBM) -> None:
        with pytest.raises(SPTInvariantError):
            MonteCarloEngine(scalar_gbm, n_steps=0)

    def test_invalid_confidence(self, scalar_gbm: CorrelatedGBM) -> None:
        with pytest.raises(SPTInvariantError):
            MonteCarloEngine(scalar_gbm, confidence_level=1.5)

    def test_result_validates(self, scalar_gbm: CorrelatedGBM) -> None:
        engine = MonteCarloEngine(scalar_gbm, n_paths=50, seed=42)
        result = engine.run()
        assert result.validate()
