"""Tests for quantspt.temporal.dynamic_covariance — DCC-GARCH.

Validates DCC-GARCH estimation, time-varying covariance output, and
the CovarianceRateProcess protocol conformance.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt._result import SPTResult
from quantspt.core.covariance import CovarianceRateProcess
from quantspt.errors import SPTInvariantError
from quantspt.temporal.dynamic_covariance import DCCGarch, DCCResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def garch_returns(rng: np.random.Generator) -> np.ndarray:
    """Generate 3-asset returns with GARCH-like volatility clustering."""
    T = 500
    n = 3
    returns = np.zeros((T, n))
    sigma2 = np.full(n, 0.0001)

    for t in range(T):
        returns[t] = rng.normal(0, np.sqrt(sigma2))
        sigma2 = 0.00001 + 0.08 * returns[t] ** 2 + 0.90 * sigma2

    return returns


@pytest.fixture()
def crisis_returns(rng: np.random.Generator) -> np.ndarray:
    """Returns with a clear volatility regime shift in the middle."""
    n = 3
    calm = rng.standard_normal((200, n)) * 0.01
    crisis = rng.standard_normal((200, n)) * 0.04
    crisis += 0.02 * rng.standard_normal((200, 1))
    return np.vstack([calm, crisis])


# ---------------------------------------------------------------------------
# DCC-GARCH Estimation
# ---------------------------------------------------------------------------


class TestDCCGarch:
    def test_result_type(self, garch_returns: np.ndarray) -> None:
        model = DCCGarch()
        result = model.fit(garch_returns)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, DCCResult)

    def test_output_shapes(self, garch_returns: np.ndarray) -> None:
        T, n = garch_returns.shape
        model = DCCGarch()
        result = model.fit(garch_returns)
        assert result.data.covariances.shape == (T, n, n)
        assert result.data.correlations.shape == (T, n, n)
        assert result.data.conditional_vols.shape == (T, n)

    def test_covariance_psd(self, garch_returns: np.ndarray) -> None:
        """All time-varying covariance matrices should be PSD."""
        model = DCCGarch()
        result = model.fit(garch_returns)
        for t in range(result.data.covariances.shape[0]):
            eigvals = np.linalg.eigvalsh(result.data.covariances[t])
            assert np.all(eigvals >= -1e-10), f"Covariance at t={t} not PSD"

    def test_correlation_bounded(self, garch_returns: np.ndarray) -> None:
        """Correlations should be in [-1, 1]."""
        model = DCCGarch()
        result = model.fit(garch_returns)
        assert np.all(result.data.correlations >= -1.0 - 1e-8)
        assert np.all(result.data.correlations <= 1.0 + 1e-8)

    def test_correlation_diagonal_one(self, garch_returns: np.ndarray) -> None:
        model = DCCGarch()
        result = model.fit(garch_returns)
        for t in range(result.data.correlations.shape[0]):
            assert_allclose(np.diag(result.data.correlations[t]), 1.0, atol=1e-6)

    def test_dcc_params_bounded(self, garch_returns: np.ndarray) -> None:
        """DCC parameters a, b should satisfy a > 0, b > 0, a + b < 1."""
        model = DCCGarch()
        result = model.fit(garch_returns)
        assert result.data.dcc_a > 0
        assert result.data.dcc_b > 0
        assert result.data.dcc_a + result.data.dcc_b < 1.0

    def test_crisis_correlations_increase(self, crisis_returns: np.ndarray) -> None:
        """Correlations should be higher during the crisis period."""
        model = DCCGarch()
        result = model.fit(crisis_returns)
        corrs = result.data.correlations

        calm_avg = np.mean(
            [np.mean(np.abs(corrs[t][np.triu_indices(3, 1)])) for t in range(50, 150)]
        )
        crisis_avg = np.mean(
            [np.mean(np.abs(corrs[t][np.triu_indices(3, 1)])) for t in range(250, 350)]
        )
        assert crisis_avg > calm_avg * 0.5, (
            "Crisis correlations should be meaningfully different"
        )

    def test_garch_params_recorded(self, garch_returns: np.ndarray) -> None:
        model = DCCGarch()
        result = model.fit(garch_returns)
        assert len(result.data.garch_params) == garch_returns.shape[1]
        for p in result.data.garch_params:
            assert "omega" in p
            assert "alpha" in p
            assert "beta" in p
            assert p["alpha"] + p["beta"] < 1.0

    def test_conditional_vols_positive(self, garch_returns: np.ndarray) -> None:
        model = DCCGarch()
        result = model.fit(garch_returns)
        assert np.all(result.data.conditional_vols > 0)

    def test_metadata(self, garch_returns: np.ndarray) -> None:
        model = DCCGarch()
        result = model.fit(garch_returns)
        assert result.metadata["method"] == "DCC-GARCH(1,1)"
        assert "persistence" in result.metadata
        assert result.computation_time_ms > 0


# ---------------------------------------------------------------------------
# CovarianceRateProcess Protocol
# ---------------------------------------------------------------------------


class TestCovarianceRateProcess:
    def test_protocol_conformance(self, garch_returns: np.ndarray) -> None:
        """DCCGarch should satisfy CovarianceRateProcess protocol."""
        model = DCCGarch()
        model.fit(garch_returns)
        assert isinstance(model, CovarianceRateProcess)

    def test_covariance_at(self, garch_returns: np.ndarray) -> None:
        model = DCCGarch()
        model.fit(garch_returns)
        cov = model.covariance_at(100.0)
        n = garch_returns.shape[1]
        assert cov.shape == (n, n)
        assert np.allclose(cov, cov.T)

    def test_n_assets(self, garch_returns: np.ndarray) -> None:
        model = DCCGarch()
        model.fit(garch_returns)
        assert model.n_assets() == garch_returns.shape[1]

    def test_covariance_at_before_fit(self) -> None:
        model = DCCGarch()
        with pytest.raises(SPTInvariantError, match="fit"):
            model.covariance_at(0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestDCCInternalFallback:
    def test_internal_garch_fallback(self, garch_returns: np.ndarray) -> None:
        """Test the internal GARCH estimator when use_arch=False."""
        model = DCCGarch(use_arch=False)
        result = model.fit(garch_returns)
        assert isinstance(result.data, DCCResult)
        assert result.data.covariances.shape[0] == garch_returns.shape[0]
        assert np.all(result.data.conditional_vols > 0)

    def test_internal_produces_psd(self, garch_returns: np.ndarray) -> None:
        model = DCCGarch(use_arch=False)
        result = model.fit(garch_returns)
        for t in range(0, result.data.covariances.shape[0], 50):
            eigvals = np.linalg.eigvalsh(result.data.covariances[t])
            assert np.all(eigvals >= -1e-10)

    def test_internal_dcc_params(self, garch_returns: np.ndarray) -> None:
        model = DCCGarch(use_arch=False)
        result = model.fit(garch_returns)
        assert result.data.dcc_a > 0
        assert result.data.dcc_b > 0
        assert result.data.dcc_a + result.data.dcc_b < 1.0

    def test_covariance_at_with_fallback(self, garch_returns: np.ndarray) -> None:
        model = DCCGarch(use_arch=False)
        model.fit(garch_returns)
        cov = model.covariance_at(0.0)
        assert cov.shape == (garch_returns.shape[1], garch_returns.shape[1])

    def test_pandas_input(self, garch_returns: np.ndarray) -> None:
        import pandas as pd

        df = pd.DataFrame(garch_returns, columns=["A", "B", "C"])
        model = DCCGarch()
        result = model.fit(df)
        assert result.data.covariances.shape == (len(df), 3, 3)

    def test_custom_times(self, garch_returns: np.ndarray) -> None:
        times = np.linspace(0, 10, garch_returns.shape[0])
        model = DCCGarch()
        model.fit(garch_returns, times=times)
        cov_mid = model.covariance_at(5.0)
        assert cov_mid.shape == (garch_returns.shape[1], garch_returns.shape[1])


class TestDCCValidation:
    def test_too_few_observations(self, rng: np.random.Generator) -> None:
        model = DCCGarch()
        with pytest.raises(SPTInvariantError, match="at least 10"):
            model.fit(rng.standard_normal((5, 3)))

    def test_single_asset_rejected(self, rng: np.random.Generator) -> None:
        model = DCCGarch()
        with pytest.raises(SPTInvariantError, match="at least 2"):
            model.fit(rng.standard_normal((100, 1)))

    def test_1d_input_rejected(self, rng: np.random.Generator) -> None:
        model = DCCGarch()
        with pytest.raises(SPTInvariantError, match="2-D"):
            model.fit(rng.standard_normal(100))


# ---------------------------------------------------------------------------
# Regression tests for DCC-GARCH fixes
# ---------------------------------------------------------------------------


class TestUncenteredQbar:
    """Q-bar must be the uncentered second moment (1/T) * sum(eps * eps'),
    NOT np.cov() which centers and applies Bessel's correction.

    The standard DCC estimator (Engle 2002) defines Q-bar as the sample
    average of the outer products of standardized residuals, without
    mean-centering.
    """

    def test_qbar_close_to_identity(self, garch_returns: np.ndarray) -> None:
        """When standardized residuals are iid standard normal,
        Q-bar should be close to the identity matrix."""
        model = DCCGarch()
        result = model.fit(garch_returns)
        T, _n = garch_returns.shape
        safe_vols = np.where(
            result.data.conditional_vols > 1e-15,
            result.data.conditional_vols,
            1e-15,
        )
        std_resid = garch_returns / safe_vols
        Q_bar = (std_resid.T @ std_resid) / T
        diag = np.diag(Q_bar)
        assert np.all(diag > 0.5), f"Q-bar diagonal should be near 1, got {diag}"

    def test_slsqp_respects_constraint(self, garch_returns: np.ndarray) -> None:
        """DCC parameters a + b must be strictly < 1 with SLSQP."""
        model = DCCGarch()
        result = model.fit(garch_returns)
        assert result.data.dcc_a + result.data.dcc_b < 1.0, (
            f"SLSQP should enforce a+b < 1, got "
            f"a={result.data.dcc_a}, b={result.data.dcc_b}"
        )
