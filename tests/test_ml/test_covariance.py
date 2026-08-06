"""Tests for ML covariance estimators — factor model and RMT denoising.

Validates that estimators produce symmetric PSD matrices and that
RMT denoiser correctly removes noise below the Marchenko-Pastur edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.ml.covariance import FactorModelEstimator, RMTDenoiser


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(99)


@pytest.fixture
def factor_returns(rng: np.random.Generator) -> np.ndarray:
    """Synthetic returns with 3 latent factors and 10 assets."""
    T, n, k = 500, 10, 3
    factors = rng.normal(0, 0.01, size=(T, k))
    loadings = rng.normal(0, 1, size=(n, k))
    noise = rng.normal(0, 0.002, size=(T, n))
    returns = factors @ loadings.T + noise
    return returns.astype(np.float64)


@pytest.fixture
def noisy_returns(rng: np.random.Generator) -> np.ndarray:
    """Returns with high noise-to-signal ratio (T ≈ n)."""
    T, n = 100, 80
    true_cov = np.eye(n) * 0.01
    true_cov[:3, :3] += 0.05
    L = np.linalg.cholesky(true_cov)
    returns = rng.normal(size=(T, n)) @ L.T
    return returns.astype(np.float64)


# ---------------------------------------------------------------------------
# Factor Model Tests
# ---------------------------------------------------------------------------


class TestFactorModelEstimator:
    """Tests for FactorModelEstimator."""

    def test_output_is_psd(self, factor_returns: np.ndarray) -> None:
        """Factor model covariance estimate must be PSD."""
        estimator = FactorModelEstimator(n_factors=3)
        estimator.fit(factor_returns)
        sigma = estimator.estimate()

        eigenvalues = np.linalg.eigvalsh(sigma)
        assert eigenvalues[0] >= -1e-10, (
            f"Not PSD: min eigenvalue = {eigenvalues[0]:.6e}"
        )

    def test_output_is_symmetric(self, factor_returns: np.ndarray) -> None:
        """Factor model output must be symmetric."""
        estimator = FactorModelEstimator(n_factors=3)
        estimator.fit(factor_returns)
        sigma = estimator.estimate()
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)

    def test_correct_shape(self, factor_returns: np.ndarray) -> None:
        """Output shape matches (n, n)."""
        estimator = FactorModelEstimator(n_factors=3)
        estimator.fit(factor_returns)
        sigma = estimator.estimate()
        assert sigma.shape == (10, 10)

    def test_n_factors_selection(self, factor_returns: np.ndarray) -> None:
        """Auto factor selection based on explained variance."""
        estimator = FactorModelEstimator(explained_variance_threshold=0.9)
        estimator.fit(factor_returns)
        assert estimator.n_factors >= 2
        assert estimator.n_factors <= 10

    def test_explicit_n_factors(self, factor_returns: np.ndarray) -> None:
        """Explicit factor count is respected."""
        estimator = FactorModelEstimator(n_factors=5)
        estimator.fit(factor_returns)
        assert estimator.n_factors == 5
        assert estimator.loadings.shape == (10, 5)

    def test_factor_covariance_is_identity(self, factor_returns: np.ndarray) -> None:
        """Factor covariance F is identity (PCA rotation)."""
        estimator = FactorModelEstimator(n_factors=3)
        estimator.fit(factor_returns)
        np.testing.assert_allclose(estimator.factor_covariance, np.eye(3), atol=1e-10)

    def test_idiosyncratic_non_negative(self, factor_returns: np.ndarray) -> None:
        """Idiosyncratic variances must be non-negative."""
        estimator = FactorModelEstimator(n_factors=3)
        estimator.fit(factor_returns)
        assert np.all(estimator.idiosyncratic_variance >= 0)

    def test_n_assets_property(self, factor_returns: np.ndarray) -> None:
        """n_assets returns correct value after fit."""
        estimator = FactorModelEstimator(n_factors=3)
        estimator.fit(factor_returns)
        assert estimator.n_assets == 10

    def test_unfitted_raises(self) -> None:
        """Accessing results before fit raises RuntimeError."""
        estimator = FactorModelEstimator()
        with pytest.raises(RuntimeError, match="fitted"):
            estimator.estimate()
        with pytest.raises(RuntimeError, match="fitted"):
            _ = estimator.loadings


# ---------------------------------------------------------------------------
# RMT Denoiser Tests
# ---------------------------------------------------------------------------


class TestRMTDenoiser:
    """Tests for RMTDenoiser."""

    def test_denoised_is_psd(self, noisy_returns: np.ndarray) -> None:
        """Denoised covariance must remain PSD."""
        denoiser = RMTDenoiser(method="constant")
        denoiser.fit(noisy_returns)
        sigma = denoiser.estimate()

        eigenvalues = np.linalg.eigvalsh(sigma)
        assert eigenvalues[0] >= -1e-10, (
            f"Not PSD: min eigenvalue = {eigenvalues[0]:.6e}"
        )

    def test_denoised_is_symmetric(self, noisy_returns: np.ndarray) -> None:
        """Denoised matrix must be symmetric."""
        denoiser = RMTDenoiser(method="constant")
        denoiser.fit(noisy_returns)
        sigma = denoiser.estimate()
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)

    def test_removes_noise_eigenvalues(self, noisy_returns: np.ndarray) -> None:
        """Eigenvalues below MP edge should be shrunk/replaced."""
        denoiser = RMTDenoiser(method="constant")
        denoiser.fit(noisy_returns)
        sigma = denoiser.estimate()

        eigenvalues = np.linalg.eigvalsh(sigma)
        noise_eigs = eigenvalues[eigenvalues < denoiser.mp_edge]
        if len(noise_eigs) > 1:
            assert np.std(noise_eigs) < 1e-10, (
                "Noise eigenvalues should be constant after denoising"
            )

    def test_mp_edge_positive(self, noisy_returns: np.ndarray) -> None:
        """Marchenko-Pastur edge must be positive."""
        denoiser = RMTDenoiser(method="constant")
        denoiser.fit(noisy_returns)
        assert denoiser.mp_edge > 0

    def test_mp_edge_formula(self, rng: np.random.Generator) -> None:
        """Verify MP edge matches theoretical formula: σ²(1 + √(n/T))²."""
        T, n = 200, 50
        returns = rng.normal(0, 0.1, size=(T, n)).astype(np.float64)
        denoiser = RMTDenoiser()
        denoiser.fit(returns)

        sigma_sq = 0.1**2
        expected_edge = sigma_sq * (1 + np.sqrt(n / T)) ** 2
        assert abs(denoiser.mp_edge - expected_edge) / expected_edge < 0.5

    def test_shrink_method(self, noisy_returns: np.ndarray) -> None:
        """Shrink method also produces PSD result."""
        denoiser = RMTDenoiser(method="shrink")
        denoiser.fit(noisy_returns)
        sigma = denoiser.estimate()

        eigenvalues = np.linalg.eigvalsh(sigma)
        assert eigenvalues[0] >= -1e-10

    def test_preserves_signal_eigenvalues(self, noisy_returns: np.ndarray) -> None:
        """Signal eigenvalues above MP edge should be preserved."""
        denoiser = RMTDenoiser(method="constant")
        denoiser.fit(noisy_returns)
        sigma_denoised = denoiser.estimate()

        centered = noisy_returns - noisy_returns.mean(axis=0)
        T = noisy_returns.shape[0]
        sigma_sample = (centered.T @ centered) / (T - 1)

        eig_sample = np.sort(np.linalg.eigvalsh(sigma_sample))[::-1]
        eig_denoised = np.sort(np.linalg.eigvalsh(sigma_denoised))[::-1]

        signal_mask = eig_sample > denoiser.mp_edge
        n_signal = signal_mask.sum()
        if n_signal > 0:
            np.testing.assert_allclose(
                eig_denoised[:n_signal], eig_sample[signal_mask], rtol=1e-5
            )

    def test_invalid_method_raises(self) -> None:
        """Invalid method string raises ValueError."""
        with pytest.raises(ValueError, match="method"):
            RMTDenoiser(method="invalid")

    def test_unfitted_raises(self) -> None:
        """Accessing results before fit raises RuntimeError."""
        denoiser = RMTDenoiser()
        with pytest.raises(RuntimeError, match="fitted"):
            denoiser.estimate()

    def test_correct_shape(self, noisy_returns: np.ndarray) -> None:
        """Output shape matches (n, n)."""
        denoiser = RMTDenoiser()
        denoiser.fit(noisy_returns)
        sigma = denoiser.estimate()
        assert sigma.shape == (80, 80)
