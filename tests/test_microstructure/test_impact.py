"""Tests for quantspt.microstructure.impact — Kyle's lambda and Almgren-Chriss.

Validates price impact estimation against known properties and the
Almgren-Chriss trajectory against the analytical solution.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt._result import SPTResult
from quantspt.backtesting.execution import ExecutionModel, ExecutionResult
from quantspt.errors import SPTInvariantError
from quantspt.microstructure.impact import (
    AlmgrenChrissExecution,
    AlmgrenChrissResult,
    KyleLambdaResult,
    estimate_kyle_lambda,
    optimal_execution_trajectory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def kyle_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic data obeying ΔP = 0.5 Q + ε."""
    T = 1000
    true_lambda = 0.5
    Q = rng.standard_normal(T)
    noise = rng.standard_normal(T) * 0.1
    dP = true_lambda * Q + noise
    return dP, Q


@pytest.fixture()
def noisy_kyle_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Kyle data with high noise (low R²)."""
    T = 500
    Q = rng.standard_normal(T) * 100
    noise = rng.standard_normal(T) * 50
    dP = 0.001 * Q + noise
    return dP, Q


# ---------------------------------------------------------------------------
# Kyle's Lambda Tests
# ---------------------------------------------------------------------------


class TestKyleLambda:
    def test_result_type(self, kyle_data: tuple) -> None:
        dP, Q = kyle_data
        result = estimate_kyle_lambda(dP, Q)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, KyleLambdaResult)

    def test_lambda_recovery(self, kyle_data: tuple) -> None:
        """Should recover true λ = 0.5 from clean data."""
        dP, Q = kyle_data
        result = estimate_kyle_lambda(dP, Q)
        assert_allclose(result.data.kyle_lambda, 0.5, atol=0.05)

    def test_intercept_near_zero(self, kyle_data: tuple) -> None:
        dP, Q = kyle_data
        result = estimate_kyle_lambda(dP, Q)
        assert abs(result.data.intercept) < 0.05

    def test_high_r_squared(self, kyle_data: tuple) -> None:
        dP, Q = kyle_data
        result = estimate_kyle_lambda(dP, Q)
        assert result.data.r_squared > 0.8

    def test_significant_t_stat(self, kyle_data: tuple) -> None:
        dP, Q = kyle_data
        result = estimate_kyle_lambda(dP, Q)
        assert abs(result.data.t_statistic) > 2.0

    def test_market_depth_inverse(self, kyle_data: tuple) -> None:
        dP, Q = kyle_data
        result = estimate_kyle_lambda(dP, Q)
        assert_allclose(
            result.data.market_depth, 1.0 / result.data.kyle_lambda, rtol=1e-10
        )

    def test_n_obs_correct(self, kyle_data: tuple) -> None:
        dP, Q = kyle_data
        result = estimate_kyle_lambda(dP, Q)
        assert result.data.n_obs == len(dP)

    def test_noisy_data_low_r2(self, noisy_kyle_data: tuple) -> None:
        dP, Q = noisy_kyle_data
        result = estimate_kyle_lambda(dP, Q)
        assert result.data.r_squared < 0.5

    def test_positive_lambda_for_positive_impact(
        self, rng: np.random.Generator
    ) -> None:
        """Buying pressure (Q > 0) pushes prices up → λ > 0."""
        T = 500
        Q = np.abs(rng.standard_normal(T))
        dP = 0.3 * Q + rng.standard_normal(T) * 0.05
        result = estimate_kyle_lambda(dP, Q)
        assert result.data.kyle_lambda > 0


# ---------------------------------------------------------------------------
# Almgren-Chriss Optimal Execution
# ---------------------------------------------------------------------------


class TestAlmgrenChriss:
    def test_result_type(self) -> None:
        result = optimal_execution_trajectory(
            total_shares=10000, T=5.0, N=50, sigma=0.02, eta=0.01
        )
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, AlmgrenChrissResult)

    def test_trajectory_boundary_conditions(self) -> None:
        """x(0) = X, x(T) = 0."""
        X = 10000.0
        result = optimal_execution_trajectory(
            total_shares=X, T=5.0, N=50, sigma=0.02, eta=0.01
        )
        assert_allclose(result.data.trajectory[0], X)
        assert_allclose(result.data.trajectory[-1], 0.0)

    def test_trajectory_monotone_decreasing(self) -> None:
        """Optimal trajectory should be monotonically decreasing."""
        result = optimal_execution_trajectory(
            total_shares=10000, T=5.0, N=50, sigma=0.02, eta=0.01
        )
        assert np.all(np.diff(result.data.trajectory) <= 1e-10)

    def test_trade_list_sums_to_total(self) -> None:
        """Sum of all trades should equal total shares."""
        X = 10000.0
        result = optimal_execution_trajectory(
            total_shares=X, T=5.0, N=50, sigma=0.02, eta=0.01
        )
        assert_allclose(result.data.trade_list.sum(), X, rtol=1e-10)

    def test_trade_list_nonnegative(self) -> None:
        """All trades should be non-negative (no buying during liquidation)."""
        result = optimal_execution_trajectory(
            total_shares=10000, T=5.0, N=50, sigma=0.02, eta=0.01
        )
        assert np.all(result.data.trade_list >= -1e-10)

    def test_analytical_trajectory_formula(self) -> None:
        r"""Verify x_k = X sinh(kappa*(T-t_k)) / sinh(kappa*T)."""
        X = 10000.0
        T = 5.0
        N = 20
        sigma = 0.02
        eta = 0.01
        gamma = 0.001
        risk_aversion = 1e-4

        result = optimal_execution_trajectory(
            total_shares=X,
            T=T,
            N=N,
            sigma=sigma,
            eta=eta,
            gamma=gamma,
            risk_aversion=risk_aversion,
        )

        tau = T / N
        eta_tilde = eta - 0.5 * gamma * tau
        kappa = np.sqrt(risk_aversion * sigma**2 / eta_tilde)

        times = np.linspace(0, T, N + 1)
        analytical = X * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)
        analytical[0] = X
        analytical[-1] = 0.0

        assert_allclose(result.data.trajectory, analytical, rtol=1e-6)

    def test_twap_when_risk_neutral(self) -> None:
        """With risk_aversion=0, trajectory should be TWAP (linear)."""
        X = 10000.0
        T = 5.0
        N = 50
        result = optimal_execution_trajectory(
            total_shares=X, T=T, N=N, sigma=0.02, eta=0.01, risk_aversion=0.0
        )
        times = np.linspace(0, T, N + 1)
        linear = X * (1 - times / T)
        assert_allclose(result.data.trajectory, linear, rtol=1e-6)

    def test_higher_risk_aversion_more_front_loaded(self) -> None:
        """Higher risk aversion → more aggressive (front-loaded) execution."""
        low_ra = optimal_execution_trajectory(
            total_shares=10000, T=5.0, N=50, sigma=0.02, eta=0.01, risk_aversion=1e-8
        )
        high_ra = optimal_execution_trajectory(
            total_shares=10000, T=5.0, N=50, sigma=0.02, eta=0.01, risk_aversion=1e-2
        )
        mid = 25
        assert high_ra.data.trajectory[mid] < low_ra.data.trajectory[mid]

    def test_expected_cost_nonnegative(self) -> None:
        result = optimal_execution_trajectory(
            total_shares=10000, T=5.0, N=50, sigma=0.02, eta=0.01
        )
        assert result.data.expected_cost >= 0

    def test_cost_variance_nonnegative(self) -> None:
        result = optimal_execution_trajectory(
            total_shares=10000, T=5.0, N=50, sigma=0.02, eta=0.01
        )
        assert result.data.cost_variance >= 0

    def test_kappa_positive(self) -> None:
        result = optimal_execution_trajectory(
            total_shares=10000, T=5.0, N=50, sigma=0.02, eta=0.01, risk_aversion=1e-4
        )
        assert result.data.kappa > 0

    def test_times_grid(self) -> None:
        N = 20
        result = optimal_execution_trajectory(
            total_shares=10000, T=5.0, N=N, sigma=0.02, eta=0.01
        )
        assert len(result.data.times) == N + 1
        assert_allclose(result.data.times[0], 0.0)
        assert_allclose(result.data.times[-1], 5.0)


# ---------------------------------------------------------------------------
# AlmgrenChrissExecution (ExecutionModel protocol)
# ---------------------------------------------------------------------------


class TestAlmgrenChrissExecution:
    def test_protocol_conformance(self) -> None:
        model = AlmgrenChrissExecution(
            eta=0.01,
            gamma=0.0,
            sigma=np.array([0.02, 0.03]),
        )
        assert isinstance(model, ExecutionModel)

    def test_execute_result(self) -> None:
        model = AlmgrenChrissExecution(
            eta=0.01,
            gamma=0.0,
            sigma=np.array([0.02, 0.03]),
        )
        result = model.execute(
            current_weights=np.array([0.5, 0.5]),
            target_weights=np.array([0.6, 0.4]),
            portfolio_value=1_000_000.0,
        )
        assert isinstance(result, ExecutionResult)
        assert_allclose(result.weights, np.array([0.6, 0.4]))
        assert result.cost >= 0

    def test_zero_trade_zero_cost(self) -> None:
        model = AlmgrenChrissExecution(
            eta=0.01,
            gamma=0.0,
            sigma=np.array([0.02, 0.03]),
        )
        result = model.execute(
            current_weights=np.array([0.5, 0.5]),
            target_weights=np.array([0.5, 0.5]),
            portfolio_value=1_000_000.0,
        )
        assert_allclose(result.cost, 0.0, atol=1e-15)

    def test_larger_trade_higher_cost(self) -> None:
        model = AlmgrenChrissExecution(
            eta=0.01,
            gamma=0.0,
            sigma=np.array([0.02, 0.03]),
        )
        small = model.execute(np.array([0.5, 0.5]), np.array([0.51, 0.49]), 1_000_000.0)
        large = model.execute(np.array([0.5, 0.5]), np.array([0.8, 0.2]), 1_000_000.0)
        assert large.cost > small.cost


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_kyle_unequal_lengths(self) -> None:
        with pytest.raises(SPTInvariantError, match="equal length"):
            estimate_kyle_lambda(np.zeros(100), np.zeros(50))

    def test_kyle_too_few_obs(self) -> None:
        with pytest.raises(SPTInvariantError, match="at least 10"):
            estimate_kyle_lambda(np.zeros(5), np.zeros(5))

    def test_ac_negative_T(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            optimal_execution_trajectory(10000, T=-1.0, N=10, sigma=0.02, eta=0.01)

    def test_ac_zero_sigma(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            optimal_execution_trajectory(10000, T=5.0, N=10, sigma=0.0, eta=0.01)

    def test_ac_negative_eta(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            optimal_execution_trajectory(10000, T=5.0, N=10, sigma=0.02, eta=-0.01)

    def test_ac_execution_zero_value(self) -> None:
        model = AlmgrenChrissExecution(
            eta=0.01,
            gamma=0.0,
            sigma=np.array([0.02]),
        )
        with pytest.raises(SPTInvariantError, match="positive"):
            model.execute(np.array([1.0]), np.array([0.5]), 0.0)


# ---------------------------------------------------------------------------
# Regression tests for Almgren-Chriss bug fixes
# ---------------------------------------------------------------------------


class TestEtaTildeFormula:
    """eta_tilde must be eta - 0.5*gamma*tau (AC2001 Eq. 18),
    NOT the hallucinated eta*(1 - 0.5*tau/T).

    The old formula made eta_tilde depend only on eta and the fraction
    tau/T, ignoring gamma entirely.  This meant permanent impact had
    no effect on the urgency parameter kappa.
    """

    def test_gamma_affects_kappa(self) -> None:
        """Non-zero gamma should change kappa (and thus the trajectory)."""
        r_no_gamma = optimal_execution_trajectory(
            10000,
            T=5.0,
            N=20,
            sigma=0.02,
            eta=0.1,
            gamma=0.0,
            risk_aversion=1e-4,
        )
        r_with_gamma = optimal_execution_trajectory(
            10000,
            T=5.0,
            N=20,
            sigma=0.02,
            eta=0.1,
            gamma=0.05,
            risk_aversion=1e-4,
        )
        assert r_no_gamma.data.kappa != r_with_gamma.data.kappa

    def test_eta_tilde_value(self) -> None:
        """Direct check: eta_tilde = eta - 0.5*gamma*tau."""
        eta, gamma, T, N = 0.1, 0.02, 5.0, 20
        tau = T / N
        expected_eta_tilde = eta - 0.5 * gamma * tau
        result = optimal_execution_trajectory(
            10000,
            T=T,
            N=N,
            sigma=0.02,
            eta=eta,
            gamma=gamma,
            risk_aversion=1e-4,
        )
        expected_kappa = np.sqrt(1e-4 * 0.02**2 / expected_eta_tilde)
        assert_allclose(result.data.kappa, expected_kappa, rtol=1e-6)


class TestCostFormula:
    """Permanent impact cost must be 0.5*gamma*X^2,
    and variance uses trajectory[k+1] (post-trade holdings).
    """

    def test_permanent_cost_independent_of_schedule(self) -> None:
        """Permanent cost = 0.5*gamma*X^2, regardless of execution speed."""
        gamma = 0.001
        X = 10000.0
        expected_perm = 0.5 * gamma * X**2

        fast = optimal_execution_trajectory(
            X,
            T=1.0,
            N=5,
            sigma=0.02,
            eta=0.01,
            gamma=gamma,
            risk_aversion=1e-2,
        )
        slow = optimal_execution_trajectory(
            X,
            T=10.0,
            N=50,
            sigma=0.02,
            eta=0.01,
            gamma=gamma,
            risk_aversion=1e-8,
        )
        assert fast.data.expected_cost >= expected_perm * 0.99
        assert slow.data.expected_cost >= expected_perm * 0.99

    def test_variance_uses_post_trade_holdings(self) -> None:
        """Variance should use trajectory[k+1], not trajectory[k].

        At step k, the trade n_k has been executed and holdings become
        x_{k+1}. The risk from period k to k+1 is proportional to
        x_{k+1}^2, not x_k^2.
        """
        result = optimal_execution_trajectory(
            10000,
            T=5.0,
            N=20,
            sigma=0.02,
            eta=0.01,
            gamma=0.0,
            risk_aversion=1e-4,
        )
        tau = 5.0 / 20
        manual_var = sum(
            0.02**2 * tau * result.data.trajectory[k + 1] ** 2 for k in range(20)
        )
        assert_allclose(result.data.cost_variance, manual_var, rtol=1e-10)


class TestAlmgrenChrissExecutionUsesACModel:
    """AlmgrenChrissExecution must use optimal_execution_trajectory(),
    not the square-root impact model eta*sigma*sqrt(shares/ADV).
    """

    def test_uses_gamma(self) -> None:
        """Execution model should accept and use gamma."""
        model = AlmgrenChrissExecution(
            eta=0.01,
            gamma=0.001,
            sigma=np.array([0.02]),
        )
        result = model.execute(np.array([1.0]), np.array([0.5]), 1_000_000.0)
        assert result.cost > 0
