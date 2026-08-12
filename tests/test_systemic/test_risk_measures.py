"""Tests for quantspt.systemic.risk_measures — CoVaR, MES, SRISK.

Validates systemic risk measures against known analytical properties
and realistic financial scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt._result import SPTResult
from quantspt.errors import SPTInvariantError
from quantspt.systemic.risk_measures import (
    CoVaRResult,
    MESResult,
    compute_covar,
    compute_delta_covar,
    compute_mes,
    compute_srisk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def correlated_returns(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """System returns and a correlated institution's returns."""
    T = 2000
    factor = rng.standard_normal(T) * 0.015
    noise_sys = rng.standard_normal(T) * 0.005
    noise_inst = rng.standard_normal(T) * 0.008
    sys_ret = factor + noise_sys
    inst_ret = 1.2 * factor + noise_inst
    return sys_ret, inst_ret


@pytest.fixture()
def independent_returns(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """System returns and an independent institution."""
    T = 2000
    sys_ret = rng.standard_normal(T) * 0.015
    inst_ret = rng.standard_normal(T) * 0.015
    return sys_ret, inst_ret


@pytest.fixture()
def panel_returns(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """System + 5-asset panel for multi-asset MES."""
    T = 1000
    sys_ret = rng.standard_normal(T) * 0.015
    assets = rng.standard_normal((T, 5)) * 0.02
    assets += 0.3 * sys_ret[:, np.newaxis]
    return sys_ret, assets


# ---------------------------------------------------------------------------
# CoVaR Tests
# ---------------------------------------------------------------------------


class TestCoVaR:
    def test_result_type(self, correlated_returns: tuple) -> None:
        sys_ret, inst_ret = correlated_returns
        result = compute_covar(sys_ret, inst_ret)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, CoVaRResult)

    def test_covar_negative_left_tail(self, correlated_returns: tuple) -> None:
        """CoVaR at 5% should be negative (left tail of system)."""
        sys_ret, inst_ret = correlated_returns
        result = compute_covar(sys_ret, inst_ret, quantile=0.05)
        assert result.data.covar < 0, "5% CoVaR should be negative"

    def test_positive_beta_for_correlated(self, correlated_returns: tuple) -> None:
        """Positively correlated institution should have positive β."""
        sys_ret, inst_ret = correlated_returns
        result = compute_covar(sys_ret, inst_ret, quantile=0.05)
        assert result.data.beta > 0

    def test_smaller_beta_for_independent(self, independent_returns: tuple) -> None:
        """Independent institution should have β closer to zero."""
        sys_ret, inst_ret = independent_returns
        result = compute_covar(sys_ret, inst_ret, quantile=0.05)
        assert abs(result.data.beta) < 0.5

    def test_var_i_at_quantile(self, correlated_returns: tuple) -> None:
        sys_ret, inst_ret = correlated_returns
        result = compute_covar(sys_ret, inst_ret, quantile=0.05)
        expected_var = np.quantile(inst_ret, 0.05)
        assert_allclose(result.data.var_i, expected_var, rtol=1e-10)

    def test_quantile_stored(self, correlated_returns: tuple) -> None:
        sys_ret, inst_ret = correlated_returns
        result = compute_covar(sys_ret, inst_ret, quantile=0.01)
        assert result.data.quantile == 0.01

    def test_conditioning_vars(
        self, correlated_returns: tuple, rng: np.random.Generator
    ) -> None:
        sys_ret, inst_ret = correlated_returns
        cond = rng.standard_normal((len(sys_ret), 2))
        result = compute_covar(sys_ret, inst_ret, conditioning_vars=cond)
        assert result.data.gamma is not None
        assert len(result.data.gamma) == 2


# ---------------------------------------------------------------------------
# ΔCoVaR Tests
# ---------------------------------------------------------------------------


class TestDeltaCoVaR:
    def test_result_type(self, correlated_returns: tuple) -> None:
        sys_ret, inst_ret = correlated_returns
        result = compute_delta_covar(sys_ret, inst_ret)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, float)

    def test_delta_negative_for_risk_contributor(
        self, correlated_returns: tuple
    ) -> None:
        """ΔCoVaR should be negative for an institution that adds systemic risk."""
        sys_ret, inst_ret = correlated_returns
        result = compute_delta_covar(sys_ret, inst_ret, quantile=0.05)
        assert result.data < 0, "ΔCoVaR should be negative for risk contributor"

    def test_delta_near_zero_for_independent(self, independent_returns: tuple) -> None:
        sys_ret, inst_ret = independent_returns
        result = compute_delta_covar(sys_ret, inst_ret, quantile=0.05)
        assert abs(result.data) < 0.01, (
            "ΔCoVaR should be near zero for independent asset"
        )


# ---------------------------------------------------------------------------
# MES Tests
# ---------------------------------------------------------------------------


class TestMES:
    def test_result_type(self, correlated_returns: tuple) -> None:
        sys_ret, inst_ret = correlated_returns
        result = compute_mes(sys_ret, inst_ret)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, MESResult)

    def test_mes_positive_loss(self, correlated_returns: tuple) -> None:
        """MES returns positive loss (negated average tail return)."""
        sys_ret, inst_ret = correlated_returns
        result = compute_mes(sys_ret, inst_ret, quantile=0.05)
        assert result.data.mes > 0, "MES should be positive (loss convention)"

    def test_var_sys_negative(self, correlated_returns: tuple) -> None:
        sys_ret, inst_ret = correlated_returns
        result = compute_mes(sys_ret, inst_ret, quantile=0.05)
        assert result.data.var_sys < 0

    def test_n_tail_obs_correct(self, correlated_returns: tuple) -> None:
        sys_ret, inst_ret = correlated_returns
        result = compute_mes(sys_ret, inst_ret, quantile=0.05)
        expected_n = int(np.sum(sys_ret <= np.quantile(sys_ret, 0.05)))
        assert result.data.n_tail_obs == expected_n

    def test_panel_mes(self, panel_returns: tuple) -> None:
        """MES for multiple assets simultaneously."""
        sys_ret, assets = panel_returns
        result = compute_mes(sys_ret, assets, quantile=0.05)
        assert result.data.component_mes is not None
        assert result.data.component_mes.shape == (5,)
        assert np.all(result.data.component_mes > 0)

    def test_higher_beta_higher_mes(self, rng: np.random.Generator) -> None:
        """An asset with higher market beta should have larger positive MES."""
        T = 5000
        market = rng.standard_normal(T) * 0.015
        low_beta = 0.5 * market + rng.standard_normal(T) * 0.01
        high_beta = 2.0 * market + rng.standard_normal(T) * 0.01
        mes_low = compute_mes(market, low_beta, quantile=0.05).data.mes
        mes_high = compute_mes(market, high_beta, quantile=0.05).data.mes
        assert mes_high > mes_low, "Higher beta -> larger positive MES"


# ---------------------------------------------------------------------------
# SRISK Tests
# ---------------------------------------------------------------------------


class TestSRISK:
    def test_result_type(self) -> None:
        result = compute_srisk(mes=0.02, book_debt=500.0, market_equity=100.0)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, float)

    def test_srisk_nonnegative(self) -> None:
        result = compute_srisk(mes=0.02, book_debt=500.0, market_equity=100.0)
        assert result.data >= 0

    def test_srisk_zero_for_well_capitalized(self) -> None:
        """A well-capitalized firm should have SRISK = 0."""
        result = compute_srisk(mes=0.001, book_debt=10.0, market_equity=1000.0)
        assert result.data == 0.0

    def test_srisk_increases_with_leverage(self) -> None:
        """Higher leverage -> higher SRISK."""
        low_lev = compute_srisk(mes=0.03, book_debt=100.0, market_equity=200.0)
        high_lev = compute_srisk(mes=0.03, book_debt=900.0, market_equity=200.0)
        assert high_lev.data >= low_lev.data

    def test_srisk_increases_with_mes(self) -> None:
        """Larger MES (positive loss) -> higher SRISK."""
        low_risk = compute_srisk(mes=0.01, book_debt=500.0, market_equity=100.0)
        high_risk = compute_srisk(mes=0.10, book_debt=500.0, market_equity=100.0)
        assert high_risk.data >= low_risk.data

    def test_lrmes_in_metadata(self) -> None:
        result = compute_srisk(mes=0.02, book_debt=500.0, market_equity=100.0)
        assert "lrmes" in result.metadata
        assert 0 <= result.metadata["lrmes"] <= 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unequal_lengths_rejected(self) -> None:
        with pytest.raises(SPTInvariantError, match="equal length"):
            compute_covar(np.zeros(100), np.zeros(50))

    def test_too_few_obs(self) -> None:
        with pytest.raises(SPTInvariantError, match="at least 30"):
            compute_covar(np.zeros(10), np.zeros(10))

    def test_invalid_quantile(self) -> None:
        with pytest.raises(SPTInvariantError, match="\\(0, 1\\)"):
            compute_covar(np.zeros(100), np.zeros(100), quantile=1.5)

    def test_negative_equity_rejected(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            compute_srisk(mes=0.02, book_debt=100.0, market_equity=-50.0)


# ---------------------------------------------------------------------------
# Regression tests for MES/SRISK bug fixes
# ---------------------------------------------------------------------------


class TestMESSignConvention:
    """MES must return POSITIVE loss, not negative crisis returns.

    The old code returned the raw average of asset returns during tail
    events (negative values). This caused LRMES = 1 - exp(-18 * mes)
    to produce log(1 + positive) > 0, which then clipped to 0 in the
    SRISK formula, making SRISK ignore MES entirely.
    """

    def test_mes_equals_negated_tail_mean(self, rng: np.random.Generator) -> None:
        """MES should equal the negated average tail return."""
        T = 10000
        sys_ret = rng.standard_normal(T) * 0.02
        asset_ret = rng.standard_normal(T) * 0.02

        result = compute_mes(sys_ret, asset_ret, quantile=0.05)
        var_sys = np.quantile(sys_ret, 0.05)
        tail_mask = sys_ret <= var_sys
        manual_mes = -asset_ret[tail_mask].mean()
        assert_allclose(result.data.mes, manual_mes, rtol=1e-10)

    def test_positive_mes_flows_to_srisk(self) -> None:
        """With positive MES, SRISK should produce meaningful shortfall
        for a leveraged firm."""
        mes = 0.05
        result = compute_srisk(mes=mes, book_debt=900.0, market_equity=100.0)
        assert result.data > 0, (
            "SRISK should be positive for leveraged firm with MES > 0"
        )
        lrmes = result.metadata["lrmes"]
        assert lrmes > 0, "LRMES should be positive when MES is positive"


class TestSRISKFormula:
    """SRISK must use distressed equity W*(1-LRMES), not current equity.

    The old formula: k*(D+W) - W*(1-LRMES) incorrectly uses current
    equity W on the right side.
    The correct formula (Brownlees & Engle 2017):
      distressed_equity = W * (1 - LRMES)
      SRISK = max(0, k*(D + distressed_equity) - distressed_equity)
    """

    def test_analytical_srisk(self) -> None:
        """Verify SRISK against hand-computed value.

        mes = 0.05, D = 900, W = 100, k = 0.08
        LRMES = 1 - exp(-18 * 0.05) = 1 - exp(-0.9) = 0.5934
        distressed_equity = 100 * (1 - 0.5934) = 40.66
        SRISK = max(0, 0.08 * (900 + 40.66) - 40.66)
             = max(0, 75.25 - 40.66) = 34.59
        """
        result = compute_srisk(mes=0.05, book_debt=900.0, market_equity=100.0)
        lrmes = 1.0 - np.exp(-18.0 * 0.05)
        dist_eq = 100.0 * (1.0 - lrmes)
        expected_srisk = max(0.0, 0.08 * (900.0 + dist_eq) - dist_eq)
        assert_allclose(result.data, expected_srisk, rtol=1e-6)

    def test_lrmes_uses_factor_18(self) -> None:
        """LRMES should use the empirical factor 1-exp(-18*MES)."""
        result = compute_srisk(mes=0.03, book_debt=500.0, market_equity=100.0)
        expected_lrmes = 1.0 - np.exp(-18.0 * 0.03)
        assert_allclose(result.metadata["lrmes"], expected_lrmes, rtol=1e-6)
