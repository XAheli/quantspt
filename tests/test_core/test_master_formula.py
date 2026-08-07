"""Exhaustive tests for core/master_formula.py.

Validates the master formula decomposition — THE central theorem of
Stochastic Portfolio Theory — against simulated market data.

Mathematical References
-----------------------
- Master formula: F&K Survey Eq. 11.2
- Drift process: F&K Survey Eq. 11.3
- Boundary term: log(G(μ_T)/G(μ_0))
- Modified entropy drift: F&K Survey Eq. 11.7, Lukacs Lectures Eq. 11.5
- Diversity generator drift: F&K Survey Remark 11.1 (Example 3)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.core.generating_functions import (
    DiversityGenerator,
    EntropyGenerator,
    ModifiedEntropyGenerator,
)
from quantspt.core.master_formula import (
    boundary_term,
    drift_integral,
    master_formula_decomposition,
    verify_master_formula,
)
from quantspt.core.processes import CorrelatedGBM, simulate_path
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_cov_2() -> np.ndarray:
    return np.array([[0.04, 0.01], [0.01, 0.09]])


@pytest.fixture()
def cov_3() -> np.ndarray:
    return np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16],
        ]
    )


def _simulate_market(
    n_assets: int,
    cov: np.ndarray,
    drifts: np.ndarray,
    x0: np.ndarray,
    T: float,
    n_steps: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate a GBM market and return (mu_path, a_path, dt)."""
    rng = np.random.default_rng(seed)
    gbm = CorrelatedGBM(mu=drifts, cov=cov, x0=x0)
    _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

    mu_path = prices / prices.sum(axis=1, keepdims=True)
    dt = T / n_steps

    a_path = np.tile(cov, (n_steps + 1, 1, 1))

    return mu_path, a_path, dt


def _compute_log_relative_return(
    prices: np.ndarray,
    mu_path: np.ndarray,
    G,
    n_steps: int,
) -> float:
    """Compute log(V^π(T) / V^μ(T)) by explicit rebalancing."""
    V_pi = 1.0
    V_mu = 1.0
    for t in range(n_steps):
        mu_t = mu_path[t]
        pi_t = G.weights(mu_t)
        returns = prices[t + 1] / prices[t]
        V_pi *= np.dot(pi_t, returns)
        V_mu *= np.dot(mu_t, returns)
    return float(np.log(V_pi / V_mu))


# =========================================================================
# 1. Boundary term
# =========================================================================


class TestBoundaryTerm:
    """Tests for log(G(μ_T)/G(μ_0)) — F&K Survey Eq. 11.2 (first term)."""

    def test_identity_when_mu_unchanged(self) -> None:
        """If μ_T = μ_0, boundary = 0."""
        mu = np.array([0.5, 0.3, 0.2])
        G = DiversityGenerator(0.5)
        assert_allclose(boundary_term(G, mu, mu), 0.0, atol=1e-14)

    def test_known_value(self) -> None:
        """Compute boundary with known G values."""
        G = DiversityGenerator(0.5)
        mu_0 = np.array([0.5, 0.5])
        mu_T = np.array([0.7, 0.3])
        expected = np.log(G(mu_T) / G(mu_0))
        assert_allclose(boundary_term(G, mu_T, mu_0), expected, atol=1e-14)

    def test_sign_diversity_concentrating(self) -> None:
        """When market concentrates, G_p decreases ⟹ boundary < 0."""
        G = DiversityGenerator(0.5)
        mu_0 = np.array([0.4, 0.3, 0.3])
        mu_T = np.array([0.8, 0.15, 0.05])
        assert boundary_term(G, mu_T, mu_0) < 0

    def test_sign_diversity_diversifying(self) -> None:
        """When market diversifies, G_p increases ⟹ boundary > 0."""
        G = DiversityGenerator(0.5)
        mu_0 = np.array([0.8, 0.15, 0.05])
        mu_T = np.array([0.4, 0.35, 0.25])
        assert boundary_term(G, mu_T, mu_0) > 0

    def test_entropy_boundary(self) -> None:
        """Boundary with EntropyGenerator."""
        G = EntropyGenerator()
        mu_0 = np.array([0.5, 0.3, 0.2])
        mu_T = np.array([0.4, 0.35, 0.25])
        expected = np.log(G(mu_T) / G(mu_0))
        assert_allclose(boundary_term(G, mu_T, mu_0), expected, atol=1e-14)

    def test_requires_positive_G(self) -> None:
        """Should raise if G evaluates to non-positive."""
        from quantspt.core.generating_functions import CustomGenerator

        G = CustomGenerator(lambda mu: -1.0, "bad")
        mu = np.array([0.5, 0.5])
        with pytest.raises(SPTInvariantError, match="positive"):
            boundary_term(G, mu, mu)


# =========================================================================
# 2. Drift integral on constant market
# =========================================================================


class TestDriftIntegralConstantMarket:
    """When μ is constant, boundary = 0 and total = drift integral."""

    def test_constant_market_boundary_zero(self) -> None:
        """If μ doesn't change, boundary term is zero."""
        n_steps = 100
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        G = DiversityGenerator(0.5)
        bnd = boundary_term(G, mu_path[-1], mu_path[0])
        assert_allclose(bnd, 0.0, atol=1e-14)

    def test_constant_market_total_equals_drift(self, cov_3: np.ndarray) -> None:
        """On constant μ: total performance = drift integral."""
        n_steps = 100
        dt = 0.01
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))

        G = DiversityGenerator(0.5)
        decomp = master_formula_decomposition(G, mu_path, a_path, dt)

        assert_allclose(decomp["boundary"], 0.0, atol=1e-14)
        assert_allclose(decomp["total"], decomp["drift_integral"], atol=1e-14)

    def test_constant_market_drift_positive(self, cov_3: np.ndarray) -> None:
        """Diversity drift on diverse constant market must be positive."""
        n_steps = 50
        dt = 0.01
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))

        G = DiversityGenerator(0.5)
        drift = drift_integral(G, mu_path, a_path, dt)
        assert drift > 0, f"Expected positive drift integral, got {drift}"

    def test_drift_integral_scales_with_time(self, cov_3: np.ndarray) -> None:
        """Doubling time horizon doubles drift integral (constant market)."""
        mu = np.array([0.5, 0.3, 0.2])
        dt = 0.01
        G = DiversityGenerator(0.5)

        n1 = 100
        mu_path_1 = np.tile(mu, (n1, 1))
        a_path_1 = np.tile(cov_3, (n1, 1, 1))
        d1 = drift_integral(G, mu_path_1, a_path_1, dt)

        n2 = 200
        mu_path_2 = np.tile(mu, (n2, 1))
        a_path_2 = np.tile(cov_3, (n2, 1, 1))
        d2 = drift_integral(G, mu_path_2, a_path_2, dt)

        assert_allclose(d2 / d1, 2.0, rtol=0.01)


# =========================================================================
# 3. Master formula identity on simulated data — THE CRITICAL TEST
# =========================================================================


class TestMasterFormulaIdentity:
    """Verify log(V^π(T)/V^μ(T)) = boundary + drift_integral.

    This is THE central theorem of SPT (F&K Survey Eq. 11.2).
    If this fails, the implementation is mathematically wrong.
    """

    @pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
    def test_3stock_gbm_diversity(self, p: float, cov_3: np.ndarray) -> None:
        """Master formula with DiversityGenerator on 3-stock GBM.

        References: F&K Survey Eq. 11.2
        """
        rng = np.random.default_rng(2024)
        n_steps = 5000
        T = 1.0
        dt = T / n_steps
        drifts = np.array([0.05, 0.08, 0.06])
        x0 = np.array([100.0, 80.0, 120.0])

        gbm = CorrelatedGBM(mu=drifts, cov=cov_3, x0=x0)
        _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

        mu_path = prices / prices.sum(axis=1, keepdims=True)
        G = DiversityGenerator(p)

        log_rel = _compute_log_relative_return(prices, mu_path, G, n_steps)

        a_path = np.tile(cov_3, (n_steps + 1, 1, 1))
        decomp = master_formula_decomposition(G, mu_path, a_path, dt)

        assert_allclose(
            log_rel,
            decomp["total"],
            atol=0.02,
            err_msg=(
                f"Master formula VIOLATED for p={p}: "
                f"actual={log_rel:.6f}, predicted={decomp['total']:.6f}, "
                f"boundary={decomp['boundary']:.6f}, "
                f"drift={decomp['drift_integral']:.6f}"
            ),
        )

    def test_3stock_gbm_entropy(self, cov_3: np.ndarray) -> None:
        """Master formula with EntropyGenerator on 3-stock GBM.

        References: F&K Survey Eq. 11.2, 11.5
        """
        rng = np.random.default_rng(999)
        n_steps = 5000
        T = 1.0
        dt = T / n_steps
        drifts = np.array([0.04, 0.06, 0.05])
        x0 = np.array([100.0, 100.0, 100.0])

        gbm = CorrelatedGBM(mu=drifts, cov=cov_3, x0=x0)
        _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

        mu_path = prices / prices.sum(axis=1, keepdims=True)
        G = EntropyGenerator()

        log_rel = _compute_log_relative_return(prices, mu_path, G, n_steps)

        a_path = np.tile(cov_3, (n_steps + 1, 1, 1))
        decomp = master_formula_decomposition(G, mu_path, a_path, dt)

        assert_allclose(
            log_rel,
            decomp["total"],
            atol=0.02,
            err_msg=(
                f"Master formula VIOLATED for Entropy: "
                f"actual={log_rel:.6f}, predicted={decomp['total']:.6f}"
            ),
        )

    def test_3stock_gbm_modified_entropy(self, cov_3: np.ndarray) -> None:
        """Master formula with ModifiedEntropyGenerator.

        References: F&K Survey Eq. 11.6-11.7
        """
        rng = np.random.default_rng(7777)
        n_steps = 5000
        T = 1.0
        dt = T / n_steps
        drifts = np.array([0.05, 0.07, 0.04])
        x0 = np.array([90.0, 110.0, 100.0])

        gbm = CorrelatedGBM(mu=drifts, cov=cov_3, x0=x0)
        _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

        mu_path = prices / prices.sum(axis=1, keepdims=True)
        G = ModifiedEntropyGenerator(c=2.0)

        log_rel = _compute_log_relative_return(prices, mu_path, G, n_steps)

        a_path = np.tile(cov_3, (n_steps + 1, 1, 1))
        decomp = master_formula_decomposition(G, mu_path, a_path, dt)

        assert_allclose(
            log_rel,
            decomp["total"],
            atol=0.02,
            err_msg=(
                f"Master formula VIOLATED for ModifiedEntropy: "
                f"actual={log_rel:.6f}, predicted={decomp['total']:.6f}"
            ),
        )

    def test_2stock_simple_market(self, simple_cov_2: np.ndarray) -> None:
        """Master formula on the simplest possible market (n=2).

        References: F&K Survey Eq. 11.2
        """
        rng = np.random.default_rng(1234)
        n_steps = 5000
        T = 1.0
        dt = T / n_steps
        drifts = np.array([0.05, 0.08])
        x0 = np.array([100.0, 100.0])

        gbm = CorrelatedGBM(mu=drifts, cov=simple_cov_2, x0=x0)
        _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

        mu_path = prices / prices.sum(axis=1, keepdims=True)
        G = DiversityGenerator(0.5)

        log_rel = _compute_log_relative_return(prices, mu_path, G, n_steps)

        a_path = np.tile(simple_cov_2, (n_steps + 1, 1, 1))
        decomp = master_formula_decomposition(G, mu_path, a_path, dt)

        assert_allclose(log_rel, decomp["total"], atol=0.02)


# =========================================================================
# 4. verify_master_formula function
# =========================================================================


class TestVerifyMasterFormula:
    """Tests for the verify_master_formula convenience function."""

    def test_returns_correct_fields(self, cov_3: np.ndarray) -> None:
        """Result dict has all expected keys."""
        n_steps = 100
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))
        G = DiversityGenerator(0.5)
        dt = 0.01

        result = verify_master_formula(
            G, mu_path, a_path, log_relative_return=0.05, dt=dt
        )

        assert "verified" in result
        assert "actual" in result
        assert "predicted" in result
        assert "boundary" in result
        assert "drift_integral" in result
        assert "error" in result
        assert "relative_error" in result

    def test_detects_matching_return(self, cov_3: np.ndarray) -> None:
        """When actual ≈ predicted, verified = True."""
        n_steps = 200
        dt = 0.01
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))
        G = DiversityGenerator(0.5)

        decomp = master_formula_decomposition(G, mu_path, a_path, dt)
        predicted = decomp["total"]

        result = verify_master_formula(
            G, mu_path, a_path, log_relative_return=predicted, dt=dt
        )
        assert result["verified"] is True
        assert_allclose(result["error"], 0.0, atol=1e-14)

    def test_detects_violation(self, cov_3: np.ndarray) -> None:
        """When actual is far from predicted, verified = False."""
        n_steps = 100
        dt = 0.01
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))
        G = DiversityGenerator(0.5)

        result = verify_master_formula(
            G, mu_path, a_path, log_relative_return=99.0, dt=dt
        )
        assert result["verified"] is False

    def test_near_zero_actual(self, cov_3: np.ndarray) -> None:
        """When actual ≈ 0, uses absolute error instead of relative."""
        n_steps = 100
        dt = 0.01
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))
        G = DiversityGenerator(0.5)

        result = verify_master_formula(
            G, mu_path, a_path, log_relative_return=1e-12, dt=dt
        )
        assert isinstance(result["relative_error"], float)

    def test_actual_and_predicted_stored(self, cov_3: np.ndarray) -> None:
        n_steps = 100
        dt = 0.01
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))
        G = DiversityGenerator(0.5)

        result = verify_master_formula(
            G, mu_path, a_path, log_relative_return=0.123, dt=dt
        )
        assert_allclose(result["actual"], 0.123)
        assert_allclose(
            result["error"],
            result["actual"] - result["predicted"],
            atol=1e-14,
        )


# =========================================================================
# 5. Drift integral validation
# =========================================================================


class TestDriftIntegral:
    """Tests for drift_integral function."""

    def test_interval_count_matches_time_points_minus_one(
        self, cov_3: np.ndarray
    ) -> None:
        """Drift integral must use N-1 intervals for N time points.

        For N+1 time points on a constant market, the integral should
        equal N * g_instant * dt (left Riemann sum over N intervals),
        NOT (N+1) * g_instant * dt.
        """
        mu = np.array([0.5, 0.3, 0.2])
        G = DiversityGenerator(0.5)
        dt = 0.01

        from quantspt.core.covariance import relative_covariance

        tau = relative_covariance(cov_3, mu)
        g_instant = G.drift(mu, tau)

        for N in [10, 50, 100, 500]:
            n_points = N + 1
            mu_path = np.tile(mu, (n_points, 1))
            a_path = np.tile(cov_3, (n_points, 1, 1))
            result = drift_integral(G, mu_path, a_path, dt)
            expected = N * g_instant * dt
            assert_allclose(
                result,
                expected,
                rtol=1e-12,
                err_msg=f"N={N}: got {result:.10f}, expected {expected:.10f}",
            )

    def test_requires_min_2_steps(self, cov_3: np.ndarray) -> None:
        """Need at least 2 time steps for integration."""
        mu_path = np.array([[0.5, 0.3, 0.2]])
        a_path = cov_3[np.newaxis, :, :]
        G = DiversityGenerator(0.5)
        with pytest.raises(SPTInvariantError, match="2 time steps"):
            drift_integral(G, mu_path, a_path, dt=0.01)

    def test_shape_mismatch_raises(self, cov_3: np.ndarray) -> None:
        """mu_path and a_path must have same number of time steps."""
        mu_path = np.tile(np.array([0.5, 0.3, 0.2]), (10, 1))
        a_path = np.tile(cov_3, (5, 1, 1))
        G = DiversityGenerator(0.5)
        with pytest.raises(SPTInvariantError, match="same length"):
            drift_integral(G, mu_path, a_path, dt=0.01)

    def test_diversity_drift_always_positive(self, cov_3: np.ndarray) -> None:
        """DiversityGenerator with p<1 always has non-negative drift.

        References: F&K Survey Remark 11.1
        """
        n_steps = 100
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))
        G = DiversityGenerator(0.5)
        d = drift_integral(G, mu_path, a_path, dt=0.01)
        assert d > 0

    def test_modified_entropy_drift_positive(self, cov_3: np.ndarray) -> None:
        """ModifiedEntropyGenerator drift = γ*_μ / H_c ≥ 0.

        References: F&K Survey Eq. 11.7
        """
        n_steps = 100
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))
        G = ModifiedEntropyGenerator(c=1.0)
        d = drift_integral(G, mu_path, a_path, dt=0.01)
        assert d > 0


# =========================================================================
# 6. Full decomposition dict structure
# =========================================================================


class TestMasterFormulaDecomposition:
    """Tests for master_formula_decomposition return structure."""

    def test_returns_correct_keys(self, cov_3: np.ndarray) -> None:
        n_steps = 50
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))
        G = DiversityGenerator(0.5)
        decomp = master_formula_decomposition(G, mu_path, a_path, dt=0.01)
        assert set(decomp.keys()) == {"boundary", "drift_integral", "total"}

    def test_total_equals_sum(self, cov_3: np.ndarray) -> None:
        """total = boundary + drift_integral."""
        n_steps = 100
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))
        G = DiversityGenerator(0.5)
        decomp = master_formula_decomposition(G, mu_path, a_path, dt=0.01)
        assert_allclose(
            decomp["total"],
            decomp["boundary"] + decomp["drift_integral"],
            atol=1e-14,
        )

    def test_all_values_finite(self, cov_3: np.ndarray) -> None:
        n_steps = 50
        mu = np.array([0.5, 0.3, 0.2])
        mu_path = np.tile(mu, (n_steps, 1))
        a_path = np.tile(cov_3, (n_steps, 1, 1))
        G = EntropyGenerator()
        decomp = master_formula_decomposition(G, mu_path, a_path, dt=0.01)
        for key, val in decomp.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"
