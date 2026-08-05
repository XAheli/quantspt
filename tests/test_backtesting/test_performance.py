"""Tests for SPT performance metrics.

Validates Sharpe ratio, max drawdown, tracking error, information ratio,
and turnover statistics against known analytical values.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from quantspt.backtesting.performance import (
    compute_performance,
    compute_turnover_stats,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    tracking_error,
)

# =========================================================================
# Sharpe ratio
# =========================================================================


class TestSharpeRatio:
    """Tests for annualized Sharpe ratio computation."""

    def test_zero_volatility_returns_zero(self) -> None:
        """Constant portfolio value → zero Sharpe."""
        values = np.ones(100)
        assert sharpe_ratio(values, dt=1 / 252) == 0.0

    def test_positive_return_positive_sharpe(self) -> None:
        """Portfolio with positive mean return and nonzero vol → positive Sharpe."""
        rng = np.random.default_rng(42)
        log_rets = rng.normal(0.0004, 0.005, 252)
        values = np.exp(np.concatenate(([0.0], np.cumsum(log_rets))))
        s = sharpe_ratio(values, dt=1 / 252)
        assert s > 0

    def test_known_sharpe_value(self) -> None:
        """Verify formula: (mean_annual_ret - rf) / annual_vol."""
        rng = np.random.default_rng(42)
        daily_ret = 0.0004
        daily_vol = 0.01
        log_returns = rng.normal(daily_ret, daily_vol, 252)
        values = np.exp(np.concatenate(([0.0], np.cumsum(log_returns))))

        s = sharpe_ratio(values, dt=1 / 252)

        ann_mean = np.mean(log_returns) * 252
        ann_vol = np.std(log_returns, ddof=1) * np.sqrt(252)
        expected = ann_mean / ann_vol
        assert_allclose(s, expected, atol=1e-10)


# =========================================================================
# Max drawdown
# =========================================================================


class TestMaxDrawdown:
    """Tests for maximum drawdown computation."""

    def test_monotone_increasing_zero_drawdown(self) -> None:
        """Monotonically increasing values → zero drawdown."""
        values = np.linspace(1.0, 2.0, 100)
        assert_allclose(max_drawdown(values), 0.0, atol=1e-14)

    def test_known_drawdown(self) -> None:
        """Portfolio goes 100 → 120 → 90 → 110.
        Max drawdown = (90 - 120) / 120 = -0.25."""
        values = np.array([100.0, 120.0, 90.0, 110.0])
        assert_allclose(max_drawdown(values), -0.25, atol=1e-14)

    def test_100_percent_loss(self) -> None:
        """Complete loss: drawdown = -1.0."""
        values = np.array([100.0, 50.0, 0.001])
        dd = max_drawdown(values)
        assert dd < -0.99

    def test_drawdown_always_nonpositive(self) -> None:
        """Drawdown can never be positive."""
        rng = np.random.default_rng(42)
        values = np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
        dd = max_drawdown(values)
        assert dd <= 0.0


# =========================================================================
# Tracking error and information ratio
# =========================================================================


class TestTrackingError:
    """Tests for tracking error computation."""

    def test_identical_portfolios_zero_tracking_error(self) -> None:
        """When portfolio = benchmark, tracking error = 0."""
        values = np.cumprod(np.concatenate(([1.0], np.full(252, 1.001))))
        track_err = tracking_error(values, values, dt=1 / 252)
        assert_allclose(track_err, 0.0, atol=1e-14)

    def test_positive_tracking_error(self) -> None:
        """Different paths should have positive tracking error."""
        rng = np.random.default_rng(42)
        port = np.exp(np.cumsum(np.concatenate(([0.0], rng.normal(0, 0.01, 252)))))
        bench = np.exp(np.cumsum(np.concatenate(([0.0], rng.normal(0, 0.01, 252)))))
        track_err = tracking_error(port, bench, dt=1 / 252)
        assert track_err > 0.0


class TestInformationRatio:
    """Tests for information ratio computation."""

    def test_identical_portfolios_zero_ir(self) -> None:
        values = np.cumprod(np.concatenate(([1.0], np.full(252, 1.001))))
        ir = information_ratio(values, values, dt=1 / 252)
        assert_allclose(ir, 0.0, atol=1e-10)

    def test_outperforming_positive_ir(self) -> None:
        """Portfolio consistently outperforming benchmark → positive IR."""
        rng = np.random.default_rng(42)
        bench = np.exp(
            np.cumsum(np.concatenate(([0.0], rng.normal(0.0003, 0.01, 252))))
        )
        port = np.exp(np.cumsum(np.concatenate(([0.0], rng.normal(0.0006, 0.01, 252)))))
        ir = information_ratio(port, bench, dt=1 / 252)
        assert ir > 0.0


# =========================================================================
# compute_performance
# =========================================================================


class TestComputePerformance:
    """Tests for the full performance metrics computation."""

    def test_all_fields_present(self) -> None:
        rng = np.random.default_rng(42)
        values = np.exp(
            np.cumsum(np.concatenate(([0.0], rng.normal(0.0003, 0.01, 252))))
        )
        metrics = compute_performance(values, dt=1 / 252)

        assert hasattr(metrics, "annualized_return")
        assert hasattr(metrics, "annualized_volatility")
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "total_return")

    def test_total_return_formula(self) -> None:
        """total_return = V_T / V_0 - 1."""
        values = np.array([1.0, 1.1, 1.2, 1.15])
        metrics = compute_performance(values, dt=1 / 252)
        assert_allclose(metrics.total_return, 0.15, atol=1e-14)

    def test_annualized_return_one_year(self) -> None:
        """Over exactly 1 year, annualized = total."""
        values = np.linspace(1.0, 1.1, 253)
        metrics = compute_performance(values, dt=1 / 252)
        assert_allclose(metrics.annualized_return, 0.1, rtol=0.01)


# =========================================================================
# compute_turnover_stats
# =========================================================================


class TestComputeTurnoverStats:
    """Tests for turnover statistics."""

    def test_avg_turnover_per_rebalance(self) -> None:
        turnover = np.array([0.2, 0.0, 0.0, 0.0, 0.1])
        stats = compute_turnover_stats(turnover, n_rebalances=2, dt=1 / 252)
        assert_allclose(stats.total_turnover, 0.3, atol=1e-14)
        assert_allclose(stats.avg_turnover_per_rebalance, 0.15, atol=1e-14)

    def test_annualized_turnover(self) -> None:
        """Annualized = total / (n_steps * dt)."""
        turnover = np.zeros(253)
        turnover[0] = 0.5
        turnover[21] = 0.3
        stats = compute_turnover_stats(turnover, n_rebalances=2, dt=1 / 252)
        expected_ann = 0.8 / (252 / 252)
        assert_allclose(stats.annualized_turnover, expected_ann, atol=1e-10)

    def test_zero_rebalances(self) -> None:
        turnover = np.zeros(100)
        stats = compute_turnover_stats(turnover, n_rebalances=0, dt=1 / 252)
        assert stats.avg_turnover_per_rebalance == 0.0
