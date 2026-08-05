"""Tests for estimation/growth_rates — growth rate estimation.

Validates that growth rate estimates converge to the true growth rates
with sufficient data, that bias correction works, and that standard
errors are correctly computed.

Mathematical References
-----------------------
- Growth rate definition: F&K Survey Eq. 1.4-1.6
- Portfolio growth rate: F&K Survey Eq. 1.12
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.errors import SPTInvariantError
from quantspt.estimation.growth_rates import (
    estimate_growth_rates,
    rolling_growth_rates,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# =========================================================================
# A. Growth Rate Estimation
# =========================================================================


class TestEstimateGrowthRates:
    """Tests for estimate_growth_rates()."""

    def test_converges_to_true_rate(self, rng: np.random.Generator) -> None:
        """With large T, estimate should converge to true growth rate."""
        true_gamma = np.array([0.05, 0.08, -0.02])
        sigma = np.array([0.2, 0.3, 0.15])
        dt = 1.0 / 252

        T = 500000
        log_returns = np.zeros((T, 3))
        for i in range(3):
            log_returns[:, i] = rng.normal(
                true_gamma[i] * dt, sigma[i] * np.sqrt(dt), size=T
            )

        result = estimate_growth_rates(
            log_returns, frequency=252, bias_correction=False
        )
        assert_allclose(result["growth_rates"], true_gamma, atol=0.02)

    def test_bias_correction_direction(self, rng: np.random.Generator) -> None:
        """Bias-corrected estimate should be slightly higher than uncorrected."""
        log_returns = rng.standard_normal((500, 3)) * 0.01
        result_bc = estimate_growth_rates(log_returns, bias_correction=True)
        result_no = estimate_growth_rates(log_returns, bias_correction=False)
        assert np.all(result_bc["growth_rates"] >= result_no["growth_rates"] - 1e-10)

    def test_zero_returns_give_zero_rate(self) -> None:
        """Zero log-returns should give approximately zero growth rate."""
        log_returns = np.zeros((100, 2))
        result = estimate_growth_rates(log_returns, bias_correction=False)
        assert_allclose(result["growth_rates"], np.zeros(2), atol=1e-14)

    def test_standard_errors_positive(self, rng: np.random.Generator) -> None:
        """Standard errors must be strictly positive for non-degenerate data."""
        log_returns = rng.standard_normal((200, 3)) * 0.01
        result = estimate_growth_rates(log_returns)
        assert np.all(result["standard_errors"] > 0)

    def test_standard_errors_decrease_with_T(self, rng: np.random.Generator) -> None:
        """SE should scale as 1/sqrt(T)."""
        log_returns_short = rng.standard_normal((100, 2)) * 0.01
        log_returns_long = rng.standard_normal((10000, 2)) * 0.01

        se_short = estimate_growth_rates(log_returns_short)["standard_errors"]
        se_long = estimate_growth_rates(log_returns_long)["standard_errors"]
        ratio = se_short / se_long
        assert np.all(ratio > 5)

    def test_n_observations(self, rng: np.random.Generator) -> None:
        log_returns = rng.standard_normal((250, 3)) * 0.01
        result = estimate_growth_rates(log_returns)
        assert result["n_observations"] == 250

    def test_annualization_frequency(self, rng: np.random.Generator) -> None:
        """Different frequencies should scale the growth rate."""
        log_returns = rng.standard_normal((200, 2)) * 0.01
        daily = estimate_growth_rates(log_returns, frequency=252, bias_correction=False)
        weekly = estimate_growth_rates(log_returns, frequency=52, bias_correction=False)
        ratio = daily["growth_rates"] / weekly["growth_rates"]
        assert_allclose(ratio, 252.0 / 52.0, atol=1e-10)

    def test_min_observations_check(self) -> None:
        log_returns = np.zeros((5, 3))
        with pytest.raises(SPTInvariantError, match="observations"):
            estimate_growth_rates(log_returns, min_observations=10)

    def test_rejects_1d_input(self) -> None:
        with pytest.raises(SPTInvariantError, match="2-D"):
            estimate_growth_rates(np.array([0.01, 0.02]))

    def test_output_shapes(self, rng: np.random.Generator) -> None:
        log_returns = rng.standard_normal((100, 5)) * 0.01
        result = estimate_growth_rates(log_returns)
        assert result["growth_rates"].shape == (5,)
        assert result["standard_errors"].shape == (5,)


# =========================================================================
# B. Rolling Growth Rates
# =========================================================================


class TestRollingGrowthRates:
    """Tests for rolling_growth_rates()."""

    def test_output_length(self, rng: np.random.Generator) -> None:
        log_returns = rng.standard_normal((100, 3)) * 0.01
        results = rolling_growth_rates(log_returns, window=20)
        assert len(results) == 81

    def test_single_window_equals_full(self, rng: np.random.Generator) -> None:
        log_returns = rng.standard_normal((50, 2)) * 0.01
        rolling = rolling_growth_rates(log_returns, window=50, bias_correction=False)
        full = estimate_growth_rates(log_returns, bias_correction=False)
        assert len(rolling) == 1
        assert_allclose(rolling[0]["growth_rates"], full["growth_rates"], atol=1e-12)

    def test_tracks_regime_change(self) -> None:
        """Rolling estimates should adapt to changing growth rates."""
        rng = np.random.default_rng(99)
        dt = 1.0 / 252
        T1, T2 = 500, 500
        ret_low = rng.normal(-0.10 * dt, 0.1 * np.sqrt(dt), (T1, 1))
        ret_high = rng.normal(0.30 * dt, 0.1 * np.sqrt(dt), (T2, 1))
        log_returns = np.vstack([ret_low, ret_high])

        results = rolling_growth_rates(log_returns, window=200)
        early_rate = results[0]["growth_rates"][0]
        late_rate = results[-1]["growth_rates"][0]
        assert late_rate > early_rate

    def test_window_too_large(self, rng: np.random.Generator) -> None:
        log_returns = rng.standard_normal((10, 2))
        with pytest.raises(SPTInvariantError, match="observations"):
            rolling_growth_rates(log_returns, window=20)
