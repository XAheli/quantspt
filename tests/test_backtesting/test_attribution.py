"""Tests for master formula attribution.

THE CRITICAL TEST: verifies that the master formula identity holds
when decomposing backtest performance into boundary + drift terms.

Mathematical References
-----------------------
- Master formula: F&K Survey Eq. 11.2
    log(V^π(T) / V^μ(T)) = log(G(μ(T))/G(μ(0))) + ∫₀ᵀ g(t) dt
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.backtesting.attribution import AttributionResult, compute_attribution
from quantspt.core.generating_functions import (
    DiversityGenerator,
    EntropyGenerator,
    ModifiedEntropyGenerator,
)
from quantspt.core.processes import CorrelatedGBM, simulate_path

# =========================================================================
# Helpers
# =========================================================================


def _simulate_and_compute_log_relative(
    G,
    cov: np.ndarray,
    drifts: np.ndarray,
    x0: np.ndarray,
    T: float,
    n_steps: int,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    """Simulate GBM, compute market weights and log-relative return.

    Returns (market_weights, log_relative_return, dt).
    """
    rng = np.random.default_rng(seed)
    gbm = CorrelatedGBM(mu=drifts, cov=cov, x0=x0)
    _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

    mu_path = prices / prices.sum(axis=1, keepdims=True)
    dt = T / n_steps

    V_pi = 1.0
    V_mu = 1.0
    for t in range(n_steps):
        pi_t = G.weights(mu_path[t])
        ret = prices[t + 1] / prices[t]
        V_pi *= float(np.dot(pi_t, ret))
        V_mu *= float(np.dot(mu_path[t], ret))

    log_rel = float(np.log(V_pi / V_mu))
    return mu_path, log_rel, dt


# =========================================================================
# Attribution result structure
# =========================================================================


class TestAttributionResult:
    """Tests for the AttributionResult data structure."""

    def test_result_fields(self) -> None:
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        G = DiversityGenerator(0.5)
        mu_path, log_rel, dt = _simulate_and_compute_log_relative(
            G,
            cov,
            np.array([0.05, 0.08]),
            np.array([100.0, 100.0]),
            1.0,
            500,
            seed=42,
        )
        result = compute_attribution(G, mu_path, cov, log_rel, dt)

        assert isinstance(result, AttributionResult)
        assert np.isfinite(result.boundary)
        assert np.isfinite(result.drift_integral)
        assert np.isfinite(result.predicted_log_relative)
        assert np.isfinite(result.residual)
        assert len(result.boundary_series) == len(mu_path)
        assert len(result.drift_series) == len(mu_path)

    def test_predicted_equals_boundary_plus_drift(self) -> None:
        """predicted = boundary + drift_integral."""
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        G = DiversityGenerator(0.5)
        mu_path, log_rel, dt = _simulate_and_compute_log_relative(
            G,
            cov,
            np.array([0.05, 0.08]),
            np.array([100.0, 100.0]),
            1.0,
            500,
            seed=42,
        )
        result = compute_attribution(G, mu_path, cov, log_rel, dt)
        assert_allclose(
            result.predicted_log_relative,
            result.boundary + result.drift_integral,
            atol=1e-14,
        )


# =========================================================================
# Master formula identity — THE CRITICAL TEST
# =========================================================================


class TestMasterFormulaIdentity:
    """Verify master formula: actual ≈ boundary + drift.

    This is the most important test in the entire backtesting module.
    If it fails, either the backtest engine or the attribution is wrong.

    References: F&K Survey Eq. 11.2
    """

    @pytest.fixture()
    def cov_3(self) -> np.ndarray:
        return np.array(
            [
                [0.04, 0.01, 0.005],
                [0.01, 0.09, 0.02],
                [0.005, 0.02, 0.16],
            ]
        )

    @pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
    def test_diversity_generator(self, p: float, cov_3: np.ndarray) -> None:
        """Master formula holds for DiversityGenerator with various p."""
        G = DiversityGenerator(p)
        drifts = np.array([0.05, 0.08, 0.06])
        x0 = np.array([100.0, 80.0, 120.0])

        mu_path, log_rel, dt = _simulate_and_compute_log_relative(
            G,
            cov_3,
            drifts,
            x0,
            T=1.0,
            n_steps=5000,
            seed=2024,
        )
        result = compute_attribution(G, mu_path, cov_3, log_rel, dt)

        assert_allclose(
            result.actual_log_relative,
            result.predicted_log_relative,
            atol=0.025,
            err_msg=(
                f"Master formula VIOLATED for Diversity(p={p}): "
                f"actual={result.actual_log_relative:.6f}, "
                f"predicted={result.predicted_log_relative:.6f}, "
                f"residual={result.residual:.6f}"
            ),
        )

    def test_entropy_generator(self, cov_3: np.ndarray) -> None:
        """Master formula holds for EntropyGenerator."""
        G = EntropyGenerator()
        drifts = np.array([0.04, 0.06, 0.05])
        x0 = np.array([100.0, 100.0, 100.0])

        mu_path, log_rel, dt = _simulate_and_compute_log_relative(
            G,
            cov_3,
            drifts,
            x0,
            T=1.0,
            n_steps=5000,
            seed=999,
        )
        result = compute_attribution(G, mu_path, cov_3, log_rel, dt)

        assert_allclose(
            result.actual_log_relative,
            result.predicted_log_relative,
            atol=0.025,
            err_msg=(
                f"Master formula VIOLATED for Entropy: "
                f"actual={result.actual_log_relative:.6f}, "
                f"predicted={result.predicted_log_relative:.6f}"
            ),
        )

    def test_modified_entropy_generator(self, cov_3: np.ndarray) -> None:
        """Master formula holds for ModifiedEntropyGenerator."""
        G = ModifiedEntropyGenerator(c=2.0)
        drifts = np.array([0.05, 0.07, 0.04])
        x0 = np.array([90.0, 110.0, 100.0])

        mu_path, log_rel, dt = _simulate_and_compute_log_relative(
            G,
            cov_3,
            drifts,
            x0,
            T=1.0,
            n_steps=5000,
            seed=7777,
        )
        result = compute_attribution(G, mu_path, cov_3, log_rel, dt)

        assert_allclose(
            result.actual_log_relative,
            result.predicted_log_relative,
            atol=0.025,
            err_msg=(
                f"Master formula VIOLATED for ModifiedEntropy: "
                f"actual={result.actual_log_relative:.6f}, "
                f"predicted={result.predicted_log_relative:.6f}"
            ),
        )


# =========================================================================
# Constant market
# =========================================================================


class TestAttributionConstantMarket:
    """On a constant market, boundary = 0 and total = drift."""

    def test_constant_weights_boundary_zero(self) -> None:
        n_steps = 100
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps + 1, 1))
        cov = np.array(
            [
                [0.04, 0.01, 0.005],
                [0.01, 0.09, 0.02],
                [0.005, 0.02, 0.16],
            ]
        )
        G = DiversityGenerator(0.5)

        result = compute_attribution(G, mu_path, cov, 0.05, dt=0.01)
        assert_allclose(result.boundary, 0.0, atol=1e-14)

    def test_constant_weights_drift_positive(self) -> None:
        """Diversity drift on diverse constant market must be positive."""
        n_steps = 200
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps + 1, 1))
        cov = np.array(
            [
                [0.04, 0.01, 0.005],
                [0.01, 0.09, 0.02],
                [0.005, 0.02, 0.16],
            ]
        )
        G = DiversityGenerator(0.5)

        result = compute_attribution(G, mu_path, cov, 0.0, dt=0.01)
        assert result.drift_integral > 0

    def test_series_lengths_match(self) -> None:
        n_steps = 50
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps + 1, 1))
        cov = np.diag([0.04, 0.09, 0.16])
        G = DiversityGenerator(0.5)

        result = compute_attribution(G, mu_path, cov, 0.0, dt=0.01)
        assert len(result.boundary_series) == n_steps + 1
        assert len(result.drift_series) == n_steps + 1


# =========================================================================
# Boundary series monotonicity
# =========================================================================


class TestBoundarySeries:
    """Tests for the cumulative boundary term time series."""

    def test_boundary_starts_at_zero(self) -> None:
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (10, 1))
        cov = np.diag([0.04, 0.09, 0.16])
        G = DiversityGenerator(0.5)

        result = compute_attribution(G, mu_path, cov, 0.0, dt=0.01)
        assert_allclose(result.boundary_series[0], 0.0, atol=1e-14)

    def test_boundary_final_matches_total(self) -> None:
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (10, 1))
        cov = np.diag([0.04, 0.09, 0.16])
        G = DiversityGenerator(0.5)

        result = compute_attribution(G, mu_path, cov, 0.0, dt=0.01)
        assert_allclose(result.boundary_series[-1], result.boundary, atol=1e-14)
