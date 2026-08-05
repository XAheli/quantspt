"""Tests for estimation/covariance — sample and shrinkage estimators.

Validates mathematical correctness by generating data from known
covariance matrices and verifying that estimators converge to the
true values, preserve symmetry/PSD, and respect shrinkage bounds.

Mathematical References
-----------------------
- Covariance rate definition: F&K Survey Eq. 1.3
- Non-degeneracy (PSD) condition: FKK Eq. 2.3
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.errors import SPTInvariantError
from quantspt.estimation.covariance import (
    ledoit_wolf,
    oracle_approximating_shrinkage,
    rolling_sample_covariance,
    sample_covariance,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def known_cov_2() -> np.ndarray:
    """Known 2×2 covariance matrix (per-period)."""
    return np.array([[0.0004, 0.0001], [0.0001, 0.0009]])


@pytest.fixture()
def known_cov_5(rng: np.random.Generator) -> np.ndarray:
    """Known 5×5 PSD covariance matrix (per-period)."""
    L = rng.standard_normal((5, 5)) * 0.02
    return L @ L.T + np.eye(5) * 0.0001


@pytest.fixture()
def returns_from_known_cov_2(
    rng: np.random.Generator, known_cov_2: np.ndarray
) -> np.ndarray:
    """10000 observations drawn from known 2×2 covariance."""
    return rng.multivariate_normal(np.zeros(2), known_cov_2, size=10000)


@pytest.fixture()
def returns_from_known_cov_5(
    rng: np.random.Generator, known_cov_5: np.ndarray
) -> np.ndarray:
    """5000 observations drawn from known 5×5 covariance."""
    return rng.multivariate_normal(np.zeros(5), known_cov_5, size=5000)


# =========================================================================
# A. Sample Covariance
# =========================================================================


class TestSampleCovariance:
    """Tests for sample_covariance()."""

    def test_recovers_known_covariance(
        self, returns_from_known_cov_2: np.ndarray, known_cov_2: np.ndarray
    ) -> None:
        """Sample covariance converges to true covariance for large T."""
        result = sample_covariance(returns_from_known_cov_2, annualize=False)
        assert_allclose(result["raw"], known_cov_2, atol=5e-5)

    def test_annualization_factor(
        self, returns_from_known_cov_2: np.ndarray, known_cov_2: np.ndarray
    ) -> None:
        """Annualised = raw × frequency."""
        result = sample_covariance(
            returns_from_known_cov_2, annualize=True, frequency=252
        )
        assert_allclose(result["annualized"], result["raw"] * 252, atol=1e-14)

    def test_no_annualization(self, returns_from_known_cov_2: np.ndarray) -> None:
        """When annualize=False, no annualized matrix is returned."""
        result = sample_covariance(returns_from_known_cov_2, annualize=False)
        assert result["annualized"] is None

    def test_symmetry(self, returns_from_known_cov_5: np.ndarray) -> None:
        """Sample covariance must be symmetric."""
        result = sample_covariance(returns_from_known_cov_5, annualize=False)
        raw = result["raw"]
        assert_allclose(raw, raw.T, atol=1e-15)

    def test_positive_semi_definite(self, returns_from_known_cov_5: np.ndarray) -> None:
        """Sample covariance must be PSD (FKK Eq. 2.3)."""
        result = sample_covariance(returns_from_known_cov_5, annualize=False)
        eigenvalues = np.linalg.eigvalsh(result["raw"])
        assert np.all(eigenvalues >= -1e-14)

    def test_n_observations_stored(self, returns_from_known_cov_2: np.ndarray) -> None:
        result = sample_covariance(returns_from_known_cov_2, annualize=False)
        assert result["n_observations"] == 10000

    def test_min_observations_check(self, rng: np.random.Generator) -> None:
        """Raise if T < min_observations."""
        returns = rng.standard_normal((5, 3))
        with pytest.raises(SPTInvariantError, match="observations"):
            sample_covariance(returns, min_observations=10)

    def test_default_min_observations(self, rng: np.random.Generator) -> None:
        """Default min_observations = n + 1."""
        returns = rng.standard_normal((3, 5))
        with pytest.raises(SPTInvariantError, match="observations"):
            sample_covariance(returns)

    def test_single_asset(self, rng: np.random.Generator) -> None:
        """Single-asset case: 1×1 covariance."""
        returns = rng.standard_normal((100, 1))
        result = sample_covariance(returns, annualize=False)
        assert result["raw"].shape == (1, 1)
        assert result["raw"][0, 0] > 0

    def test_identity_covariance(self, rng: np.random.Generator) -> None:
        """Recover identity (up to scaling) from iid standard normal."""
        returns = rng.standard_normal((50000, 3))
        result = sample_covariance(returns, annualize=False)
        assert_allclose(result["raw"], np.eye(3), atol=0.02)

    def test_diagonal_covariance(self, rng: np.random.Generator) -> None:
        """Recover diagonal covariance from independent assets."""
        true_var = np.array([0.01, 0.04, 0.09])
        true_cov = np.diag(true_var)
        returns = rng.multivariate_normal(np.zeros(3), true_cov, size=20000)
        result = sample_covariance(returns, annualize=False)
        assert_allclose(np.diag(result["raw"]), true_var, atol=0.002)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(result["raw"][i, j]) < 0.005

    def test_custom_frequency(self, returns_from_known_cov_2: np.ndarray) -> None:
        """Custom frequency (e.g., 52 for weekly)."""
        result = sample_covariance(
            returns_from_known_cov_2, annualize=True, frequency=52
        )
        assert_allclose(result["annualized"], result["raw"] * 52, atol=1e-14)

    def test_rejects_1d_input(self) -> None:
        with pytest.raises(SPTInvariantError, match="2-D"):
            sample_covariance(np.array([1.0, 2.0, 3.0]))


# =========================================================================
# B. Rolling Sample Covariance
# =========================================================================


class TestRollingSampleCovariance:
    """Tests for rolling_sample_covariance()."""

    def test_output_length(self, rng: np.random.Generator) -> None:
        """Number of windows = T - window + 1."""
        returns = rng.standard_normal((100, 3))
        results = rolling_sample_covariance(returns, window=20, annualize=False)
        assert len(results) == 100 - 20 + 1

    def test_each_window_is_valid(self, rng: np.random.Generator) -> None:
        """Each rolling estimate should be symmetric PSD."""
        returns = rng.standard_normal((50, 2))
        results = rolling_sample_covariance(returns, window=15, annualize=False)
        for res in results:
            raw = res["raw"]
            assert_allclose(raw, raw.T, atol=1e-14)
            eigs = np.linalg.eigvalsh(raw)
            assert np.all(eigs >= -1e-14)

    def test_single_window_equals_full(self, rng: np.random.Generator) -> None:
        """Rolling with window=T should match full-sample estimate."""
        returns = rng.standard_normal((30, 2))
        rolling = rolling_sample_covariance(returns, window=30, annualize=False)
        full = sample_covariance(returns, annualize=False)
        assert len(rolling) == 1
        assert_allclose(rolling[0]["raw"], full["raw"], atol=1e-14)

    def test_window_too_large(self, rng: np.random.Generator) -> None:
        """Raise if window > T."""
        returns = rng.standard_normal((10, 2))
        with pytest.raises(SPTInvariantError, match="observations"):
            rolling_sample_covariance(returns, window=20)

    def test_annualization(self, rng: np.random.Generator) -> None:
        returns = rng.standard_normal((50, 2))
        results = rolling_sample_covariance(returns, window=20, annualize=True)
        for res in results:
            assert_allclose(res["annualized"], res["raw"] * 252, atol=1e-14)

    def test_rolling_adapts_to_regime(self) -> None:
        """Rolling covariance should track changing volatility regimes."""
        rng = np.random.default_rng(123)
        low_vol = rng.standard_normal((100, 2)) * 0.01
        high_vol = rng.standard_normal((100, 2)) * 0.10
        returns = np.vstack([low_vol, high_vol])

        results = rolling_sample_covariance(returns, window=50, annualize=False)
        early_vol = results[0]["raw"][0, 0]
        late_vol = results[-1]["raw"][0, 0]
        assert late_vol > early_vol * 10

    def test_min_periods_below_window(self, rng: np.random.Generator) -> None:
        """min_periods < window should still work."""
        returns = rng.standard_normal((50, 2))
        results = rolling_sample_covariance(
            returns, window=20, min_periods=5, annualize=False
        )
        assert len(results) == 31


# =========================================================================
# C. Ledoit-Wolf Shrinkage
# =========================================================================


class TestLedoitWolf:
    """Tests for ledoit_wolf() shrinkage estimator."""

    def test_intensity_in_unit_interval(
        self, returns_from_known_cov_5: np.ndarray
    ) -> None:
        """Shrinkage intensity α must be in [0, 1]."""
        result = ledoit_wolf(returns_from_known_cov_5, annualize=False)
        alpha = result["shrinkage_intensity"]
        assert 0.0 <= alpha <= 1.0

    def test_result_is_convex_combination(
        self, returns_from_known_cov_5: np.ndarray
    ) -> None:
        r"""Verify Σ_shrunk = α·F + (1−α)·S."""
        result = ledoit_wolf(returns_from_known_cov_5, annualize=False)
        alpha = result["shrinkage_intensity"]
        S = result["sample_covariance"]
        F = result["shrinkage_target"]
        expected = alpha * F + (1.0 - alpha) * S
        assert_allclose(result["raw"], expected, atol=1e-14)

    def test_shrinkage_target_is_scaled_identity(
        self, returns_from_known_cov_5: np.ndarray
    ) -> None:
        """Target F = trace(S)/n · I."""
        result = ledoit_wolf(returns_from_known_cov_5, annualize=False)
        S = result["sample_covariance"]
        n = S.shape[0]
        expected = np.trace(S) / n * np.eye(n)
        assert_allclose(result["shrinkage_target"], expected, atol=1e-14)

    def test_symmetry(self, returns_from_known_cov_5: np.ndarray) -> None:
        result = ledoit_wolf(returns_from_known_cov_5, annualize=False)
        cov = result["covariance"]
        assert_allclose(cov, cov.T, atol=1e-14)

    def test_positive_definite(self, returns_from_known_cov_5: np.ndarray) -> None:
        """Shrinkage toward identity guarantees PD (FKK Eq. 2.3)."""
        result = ledoit_wolf(returns_from_known_cov_5, annualize=False)
        eigenvalues = np.linalg.eigvalsh(result["covariance"])
        assert np.all(eigenvalues > -1e-14)

    def test_annualization(self, returns_from_known_cov_5: np.ndarray) -> None:
        result = ledoit_wolf(returns_from_known_cov_5, annualize=True)
        assert_allclose(result["covariance"], result["raw"] * 252, atol=1e-14)

    def test_converges_to_sample_with_many_observations(
        self, rng: np.random.Generator
    ) -> None:
        """With T >> n, shrinkage should be minimal (α → 0)."""
        true_cov = np.array([[0.01, 0.005], [0.005, 0.02]])
        returns = rng.multivariate_normal(np.zeros(2), true_cov, size=50000)
        result = ledoit_wolf(returns, annualize=False)
        assert result["shrinkage_intensity"] < 0.05

    def test_high_shrinkage_for_few_observations(
        self, rng: np.random.Generator
    ) -> None:
        """With T ≈ n, shrinkage should be substantial (α → 1)."""
        returns = rng.standard_normal((12, 10))
        result = ledoit_wolf(returns, annualize=False)
        assert result["shrinkage_intensity"] > 0.3

    def test_closer_to_truth_than_sample(self, rng: np.random.Generator) -> None:
        """Shrinkage estimate should have lower Frobenius error vs truth."""
        true_cov = np.diag([0.01, 0.02, 0.03, 0.04, 0.05])
        returns = rng.multivariate_normal(np.zeros(5), true_cov, size=30)
        result = ledoit_wolf(returns, annualize=False)

        err_shrunk = float(np.sqrt(np.sum((result["raw"] - true_cov) ** 2)))
        err_sample = float(
            np.sqrt(np.sum((result["sample_covariance"] - true_cov) ** 2))
        )
        assert err_shrunk <= err_sample * 1.1

    def test_min_observations(self, rng: np.random.Generator) -> None:
        returns = rng.standard_normal((3, 5))
        with pytest.raises(SPTInvariantError, match="observations"):
            ledoit_wolf(returns)

    def test_identical_returns_high_shrinkage(self) -> None:
        """Degenerate data (identical rows) -> high shrinkage."""
        returns = np.tile([0.01, -0.02, 0.03], (20, 1))
        returns = (
            returns + np.random.default_rng(99).standard_normal(returns.shape) * 1e-10
        )
        result = ledoit_wolf(returns, annualize=False)
        assert result["shrinkage_intensity"] > 0.5


# =========================================================================
# D. Oracle Approximating Shrinkage
# =========================================================================


class TestOracleApproximatingShrinkage:
    """Tests for oracle_approximating_shrinkage()."""

    def test_intensity_in_unit_interval(
        self, returns_from_known_cov_5: np.ndarray
    ) -> None:
        result = oracle_approximating_shrinkage(
            returns_from_known_cov_5, annualize=False
        )
        alpha = result["shrinkage_intensity"]
        assert 0.0 <= alpha <= 1.0

    def test_result_is_convex_combination(
        self, returns_from_known_cov_5: np.ndarray
    ) -> None:
        result = oracle_approximating_shrinkage(
            returns_from_known_cov_5, annualize=False
        )
        alpha = result["shrinkage_intensity"]
        S = result["sample_covariance"]
        F = result["shrinkage_target"]
        expected = alpha * F + (1.0 - alpha) * S
        assert_allclose(result["raw"], expected, atol=1e-14)

    def test_symmetry(self, returns_from_known_cov_5: np.ndarray) -> None:
        result = oracle_approximating_shrinkage(
            returns_from_known_cov_5, annualize=False
        )
        cov = result["covariance"]
        assert_allclose(cov, cov.T, atol=1e-14)

    def test_positive_definite(self, returns_from_known_cov_5: np.ndarray) -> None:
        result = oracle_approximating_shrinkage(
            returns_from_known_cov_5, annualize=False
        )
        eigenvalues = np.linalg.eigvalsh(result["covariance"])
        assert np.all(eigenvalues > -1e-14)

    def test_annualization(self, returns_from_known_cov_5: np.ndarray) -> None:
        result = oracle_approximating_shrinkage(
            returns_from_known_cov_5, annualize=True
        )
        assert_allclose(result["covariance"], result["raw"] * 252, atol=1e-14)

    def test_low_shrinkage_large_sample(self, rng: np.random.Generator) -> None:
        """With T >> n, OAS intensity → 0."""
        true_cov = np.diag([0.01, 0.02])
        returns = rng.multivariate_normal(np.zeros(2), true_cov, size=50000)
        result = oracle_approximating_shrinkage(returns, annualize=False)
        assert result["shrinkage_intensity"] < 0.05

    def test_high_shrinkage_small_sample(self, rng: np.random.Generator) -> None:
        """With T ≈ n, OAS intensity should be substantial."""
        returns = rng.standard_normal((12, 10))
        result = oracle_approximating_shrinkage(returns, annualize=False)
        assert result["shrinkage_intensity"] > 0.3

    def test_closer_to_truth_than_sample(self, rng: np.random.Generator) -> None:
        """OAS should reduce Frobenius error vs sample covariance."""
        true_cov = np.diag([0.01, 0.02, 0.03, 0.04, 0.05])
        returns = rng.multivariate_normal(np.zeros(5), true_cov, size=30)
        result = oracle_approximating_shrinkage(returns, annualize=False)

        err_shrunk = float(np.sqrt(np.sum((result["raw"] - true_cov) ** 2)))
        err_sample = float(
            np.sqrt(np.sum((result["sample_covariance"] - true_cov) ** 2))
        )
        assert err_shrunk <= err_sample * 1.1

    def test_min_observations(self, rng: np.random.Generator) -> None:
        returns = rng.standard_normal((3, 5))
        with pytest.raises(SPTInvariantError, match="observations"):
            oracle_approximating_shrinkage(returns)


# =========================================================================
# E. Cross-estimator comparison
# =========================================================================


class TestCrossEstimatorComparison:
    """Compare sample, LW, and OAS estimators on the same data."""

    def test_all_agree_for_large_sample(self, rng: np.random.Generator) -> None:
        """All estimators should converge for T >> n."""
        true_cov = np.array([[0.01, 0.005], [0.005, 0.02]])
        returns = rng.multivariate_normal(np.zeros(2), true_cov, size=50000)

        s = sample_covariance(returns, annualize=False)
        lw = ledoit_wolf(returns, annualize=False)
        oas = oracle_approximating_shrinkage(returns, annualize=False)

        assert_allclose(s["raw"], true_cov, atol=5e-4)
        assert_allclose(lw["raw"], true_cov, atol=5e-4)
        assert_allclose(oas["raw"], true_cov, atol=5e-4)

    def test_shrinkage_improves_condition_number(
        self, rng: np.random.Generator
    ) -> None:
        """Shrinkage should improve conditioning (lower condition number)."""
        returns = rng.standard_normal((20, 10))
        s = sample_covariance(returns, annualize=False)
        lw = ledoit_wolf(returns, annualize=False)

        cond_sample = np.linalg.cond(s["raw"])
        cond_lw = np.linalg.cond(lw["raw"])
        assert cond_lw <= cond_sample
