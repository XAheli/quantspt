"""Deep covariance estimation for Stochastic Portfolio Theory.

Provides learned covariance estimators that produce symmetric PSD matrices
compatible with the core SPT framework (relative_covariance, τ^μ, etc.).

Two approaches:
  - FactorModelEstimator: PCA-based factor model Σ = BFB' + D
  - RMTDenoiser: Marchenko-Pastur denoising of sample eigenvalues

Both produce improved estimates of the covariance rate matrix a_{ij}(t)
that drives excess growth and portfolio performance.

References
----------
Fernholz & Karatzas, "Stochastic Portfolio Theory: An Overview,"
Handbook of Numerical Analysis, 2009 (F&K Survey Eq. 1.3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..errors import SPTInvariantError


class FactorModelEstimator:
    """PCA-based factor model covariance estimator: Σ = BFB' + D.

    Decomposes the covariance structure into:
      - B: (n, k) factor loading matrix
      - F: (k, k) factor covariance matrix
      - D: (n, n) diagonal idiosyncratic noise matrix

    The resulting estimate is guaranteed to be symmetric PSD.

    Parameters
    ----------
    n_factors : int | None
        Number of factors k. If None, selected via explained variance
        threshold (retains factors explaining ≥ threshold of variance).
    explained_variance_threshold : float
        Minimum cumulative explained variance ratio. Default 0.95.

    References
    ----------
    F&K Survey Eq. 1.3 — covariance rate matrix structure.
    """

    def __init__(
        self,
        n_factors: int | None = None,
        explained_variance_threshold: float = 0.95,
    ) -> None:
        self._n_factors = n_factors
        self._threshold = explained_variance_threshold
        self._loadings: NDArray[np.float64] | None = None
        self._factor_cov: NDArray[np.float64] | None = None
        self._idiosyncratic: NDArray[np.float64] | None = None
        self._n_assets: int = 0
        self._fitted = False

    @property
    def n_assets(self) -> int:
        """Number of assets in the fitted model."""
        return self._n_assets

    @property
    def n_factors(self) -> int:
        """Number of retained factors."""
        if self._loadings is None:
            raise RuntimeError("Model must be fitted first.")
        return self._loadings.shape[1]

    @property
    def loadings(self) -> NDArray[np.float64]:
        """Factor loading matrix B of shape (n, k)."""
        if self._loadings is None:
            raise RuntimeError("Model must be fitted first.")
        return self._loadings

    @property
    def factor_covariance(self) -> NDArray[np.float64]:
        """Factor covariance matrix F of shape (k, k)."""
        if self._factor_cov is None:
            raise RuntimeError("Model must be fitted first.")
        return self._factor_cov

    @property
    def idiosyncratic_variance(self) -> NDArray[np.float64]:
        """Diagonal of idiosyncratic noise D, shape (n,)."""
        if self._idiosyncratic is None:
            raise RuntimeError("Model must be fitted first.")
        return self._idiosyncratic

    def fit(
        self,
        returns: NDArray[np.float64],
        *,
        timestamps: NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> FactorModelEstimator:
        """Fit the factor model via PCA on the return series.

        Parameters
        ----------
        returns : ndarray of shape (T, n)
            Historical return matrix.
        timestamps : array, optional
            Timestamps (unused, for protocol compliance).
        **kwargs
            Additional parameters (unused).

        Returns
        -------
        FactorModelEstimator
            The fitted model.
        """
        T, n = returns.shape
        self._n_assets = n

        centered = returns - returns.mean(axis=0)
        sample_cov = (centered.T @ centered) / max(T - 1, 1)

        eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues = np.maximum(eigenvalues, 0.0)

        if self._n_factors is not None:
            k = min(self._n_factors, n)
        else:
            total_var = eigenvalues.sum()
            if total_var < 1e-12:
                k = 1
            else:
                cumvar = np.cumsum(eigenvalues) / total_var
                k = int(np.searchsorted(cumvar, self._threshold) + 1)
                k = min(k, n)

        loadings = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])
        self._loadings = loadings
        self._factor_cov = np.eye(k, dtype=np.float64)

        reconstructed = loadings @ loadings.T
        residual_diag = np.diag(sample_cov) - np.diag(reconstructed)
        self._idiosyncratic = np.maximum(residual_diag, 0.0)

        self._fitted = True
        return self

    def estimate(
        self,
        t: int | None = None,
    ) -> NDArray[np.float64]:
        """Estimate the covariance matrix Σ = BFB' + D.

        Parameters
        ----------
        t : int, optional
            Time index (unused — model is time-invariant).

        Returns
        -------
        ndarray of shape (n, n)
            Symmetric PSD covariance matrix.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")
        loadings = self._loadings
        factor_cov = self._factor_cov
        idiosyncratic = self._idiosyncratic
        if loadings is None or factor_cov is None or idiosyncratic is None:
            raise RuntimeError("Model must be fitted first.")

        sigma: NDArray[np.float64] = loadings @ factor_cov @ loadings.T + np.diag(
            idiosyncratic
        )
        return sigma.astype(np.float64)


class RMTDenoiser:
    """Random Matrix Theory denoiser using Marchenko-Pastur distribution.

    Removes noise eigenvalues that fall below the Marchenko-Pastur upper
    edge, replacing them with a denoised estimate. The result is a
    cleaned covariance matrix that retains signal eigenvalues while
    suppressing estimation noise.

    The Marchenko-Pastur upper edge for ratio q = T/n is:
        λ_+ = σ² (1 + √(n/T))²

    Eigenvalues below λ_+ are considered noise and shrunk toward their
    average (or set to a target value).

    Parameters
    ----------
    method : str
        Denoising method:
        - 'constant': replace noise eigenvalues with their mean
        - 'shrink': shrink noise eigenvalues toward the MP mean
        Default 'constant'.

    References
    ----------
    Marchenko & Pastur, "Distribution of eigenvalues for some sets of
    random matrices," 1967.
    """

    def __init__(self, method: str = "constant") -> None:
        if method not in ("constant", "shrink"):
            raise ValueError(f"method must be 'constant' or 'shrink', got {method!r}")
        self._method = method
        self._n_assets: int = 0
        self._mp_edge: float = 0.0
        self._denoised_cov: NDArray[np.float64] | None = None
        self._fitted = False

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self._n_assets

    @property
    def mp_edge(self) -> float:
        """Marchenko-Pastur upper edge λ_+."""
        return self._mp_edge

    def fit(
        self,
        returns: NDArray[np.float64],
        *,
        timestamps: NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> RMTDenoiser:
        """Fit the RMT denoiser to return data.

        Parameters
        ----------
        returns : ndarray of shape (T, n)
            Historical return matrix.
        timestamps : array, optional
            Timestamps (unused).
        **kwargs
            Additional parameters.

        Returns
        -------
        RMTDenoiser
            The fitted denoiser.
        """
        T, n = returns.shape
        self._n_assets = n
        q = T / n

        centered = returns - returns.mean(axis=0)
        sample_cov = (centered.T @ centered) / max(T - 1, 1)

        sigma_sq = np.trace(sample_cov) / n
        self._mp_edge = sigma_sq * (1 + np.sqrt(1.0 / q)) ** 2

        eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        denoised_eigenvalues = self._denoise_eigenvalues(eigenvalues)

        cov = eigenvectors @ np.diag(denoised_eigenvalues) @ eigenvectors.T
        self._denoised_cov = (cov + cov.T) / 2.0

        eigcheck = np.linalg.eigvalsh(self._denoised_cov)  # type: ignore[arg-type]
        if eigcheck[0] < -1e-10:
            raise SPTInvariantError(
                f"Denoised covariance is not PSD: min eigenvalue = {eigcheck[0]:.4e}"
            )

        self._fitted = True
        return self

    def _denoise_eigenvalues(
        self, eigenvalues: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Replace noise eigenvalues below MP edge."""
        noise_mask = eigenvalues < self._mp_edge
        denoised = eigenvalues.copy()

        if self._method == "constant":
            if noise_mask.any():
                noise_mean = eigenvalues[noise_mask].mean()
                denoised[noise_mask] = noise_mean
        elif self._method == "shrink" and noise_mask.any():
            noise_mean = eigenvalues[noise_mask].mean()
            alpha = 0.5
            denoised[noise_mask] = (
                alpha * eigenvalues[noise_mask] + (1 - alpha) * noise_mean
            )

        return np.maximum(denoised, 0.0)

    def estimate(
        self,
        t: int | None = None,
    ) -> NDArray[np.float64]:
        """Return the denoised covariance matrix.

        Parameters
        ----------
        t : int, optional
            Time index (unused — single estimate).

        Returns
        -------
        ndarray of shape (n, n)
            Denoised symmetric PSD covariance matrix.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")
        assert self._denoised_cov is not None
        return self._denoised_cov.copy()


__all__ = [
    "FactorModelEstimator",
    "RMTDenoiser",
]
