"""Scikit-learn compatible transformers for SPT features.

Provides TransformerMixin classes that extract Stochastic Portfolio Theory
quantities (market weights, diversity measures, excess growth rates) as
features for machine learning pipelines.

Requires scikit-learn: ``pip install scikit-learn``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "DiversityFeature",
    "ExcessGrowthFeature",
    "SPTTransformer",
]


def _require_sklearn() -> Any:
    """Import sklearn or raise with installation instructions."""
    try:
        from sklearn.base import BaseEstimator, TransformerMixin

        return BaseEstimator, TransformerMixin
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for the sklearn bridge. "
            "Install with: pip install scikit-learn"
        ) from exc


class SPTTransformer:
    """Sklearn TransformerMixin that computes market weights from prices.

    Transforms a price matrix into market-capitalization weights at each
    time step. Compatible with sklearn Pipeline.

    Parameters
    ----------
    normalize : bool
        Whether to normalize weights to sum to 1 (default True).
    min_weight : float
        Minimum weight threshold. Assets below this are set to zero
        and the remaining weights are renormalized.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([[100, 200], [110, 190], [105, 210]])
    >>> transformer = SPTTransformer()
    >>> transformer.fit(prices)  # no-op for this transformer
    SPTTransformer(normalize=True, min_weight=0.0)
    >>> weights = transformer.transform(prices)
    >>> weights.shape
    (3, 2)
    """

    def __init__(
        self,
        normalize: bool = True,
        min_weight: float = 0.0,
    ) -> None:
        _require_sklearn()
        self.normalize = normalize
        self.min_weight = min_weight

    def fit(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
        y: Any = None,
    ) -> SPTTransformer:
        """Fit (no-op for this stateless transformer).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Price matrix.
        y : ignored

        Returns
        -------
        self
        """
        return self

    def transform(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
    ) -> NDArray[np.float64]:
        """Transform prices to market weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Price matrix where each row is a time step and each column
            is an asset.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Market-capitalization weights at each time step.
        """
        arr: NDArray[np.float64] = np.asarray(
            X.to_numpy(dtype=np.float64) if isinstance(X, pd.DataFrame) else X,
            dtype=np.float64,
        )

        require(arr.ndim == 2, f"Input must be 2-D, got ndim={arr.ndim}")

        row_sums = arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        weights: NDArray[np.float64] = arr / row_sums

        if self.min_weight > 0:
            weights = np.asarray(
                np.where(weights >= self.min_weight, weights, 0.0),
                dtype=np.float64,
            )
            if self.normalize:
                new_sums = weights.sum(axis=1, keepdims=True)
                new_sums = np.where(new_sums > 0, new_sums, 1.0)
                weights = weights / new_sums

        return weights

    def fit_transform(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
        y: Any = None,
    ) -> NDArray[np.float64]:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        return {"normalize": self.normalize, "min_weight": self.min_weight}

    def set_params(self, **params: Any) -> SPTTransformer:
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return (
            f"SPTTransformer(normalize={self.normalize}, min_weight={self.min_weight})"
        )


class DiversityFeature:
    """Sklearn feature extractor that computes diversity measures.

    Extracts the diversity index D_p(μ) = (Σ μ_i^p)^{1/p} at each time
    step as a single feature for ML pipelines.

    Parameters
    ----------
    p : float
        Diversity exponent, p ∈ (0, 1). Smaller p measures heavier tails.
    from_weights : bool
        If True, input X is already market weights.
        If False, X is prices and weights are computed first.

    Examples
    --------
    >>> weights = np.array([[0.5, 0.5], [0.8, 0.2], [0.3, 0.7]])
    >>> feat = DiversityFeature(p=0.5)
    >>> feat.fit_transform(weights, from_weights_input=True).shape
    (3, 1)
    """

    def __init__(
        self,
        p: float = 0.5,
        from_weights: bool = False,
    ) -> None:
        _require_sklearn()
        require(0 < p <= 1, f"Diversity parameter p must be in (0, 1], got {p}")
        self.p = p
        self.from_weights = from_weights

    def fit(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
        y: Any = None,
    ) -> DiversityFeature:
        """Fit (no-op)."""
        return self

    def transform(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
    ) -> NDArray[np.float64]:
        """Extract diversity feature from price or weight matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Price matrix or weight matrix (if from_weights=True).

        Returns
        -------
        ndarray of shape (n_samples, 1)
            Diversity index D_p at each time step.
        """
        arr: NDArray[np.float64] = np.asarray(
            X.to_numpy(dtype=np.float64) if isinstance(X, pd.DataFrame) else X,
            dtype=np.float64,
        )

        require(arr.ndim == 2, f"Input must be 2-D, got ndim={arr.ndim}")

        if self.from_weights:
            weights = arr
        else:
            row_sums = arr.sum(axis=1, keepdims=True)
            row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
            weights = arr / row_sums_safe

        weights = np.clip(weights, 1e-15, None)
        diversity = np.sum(weights**self.p, axis=1) ** (1.0 / self.p)
        return diversity.reshape(-1, 1)

    def fit_transform(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
        y: Any = None,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Fit and transform."""
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters."""
        return {"p": self.p, "from_weights": self.from_weights}

    def set_params(self, **params: Any) -> DiversityFeature:
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return f"DiversityFeature(p={self.p}, from_weights={self.from_weights})"


class ExcessGrowthFeature:
    """Sklearn feature extractor that computes excess growth rate γ*.

    Extracts the excess growth rate at each time step using a rolling
    window covariance estimate.

    Parameters
    ----------
    window : int
        Rolling window size for covariance estimation.
    from_weights : bool
        If True, input X is already market weights.
        If False, X is prices (returns are computed internally).
    min_periods : int
        Minimum observations required before computing.

    Examples
    --------
    >>> prices = np.random.default_rng(42).uniform(50, 150, (100, 5))
    >>> feat = ExcessGrowthFeature(window=20)
    >>> result = feat.fit_transform(prices)
    >>> result.shape
    (100, 1)
    """

    def __init__(
        self,
        window: int = 60,
        from_weights: bool = False,
        min_periods: int = 10,
    ) -> None:
        _require_sklearn()
        require(window >= 2, f"window must be >= 2, got {window}")
        require(min_periods >= 2, f"min_periods must be >= 2, got {min_periods}")
        self.window = window
        self.from_weights = from_weights
        self.min_periods = min_periods

    def fit(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
        y: Any = None,
    ) -> ExcessGrowthFeature:
        """Fit (no-op)."""
        return self

    def transform(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
    ) -> NDArray[np.float64]:
        """Extract excess growth rate feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Price matrix or weight matrix.

        Returns
        -------
        ndarray of shape (n_samples, 1)
            Excess growth rate γ* at each time step.
            NaN for initial periods without enough data.
        """
        arr: NDArray[np.float64] = np.asarray(
            X.to_numpy(dtype=np.float64) if isinstance(X, pd.DataFrame) else X,
            dtype=np.float64,
        )

        require(arr.ndim == 2, f"Input must be 2-D, got ndim={arr.ndim}")
        T, n = arr.shape

        log_arr = np.log(np.clip(arr, 1e-15, None))
        rets = np.vstack([np.full((1, n), np.nan), np.diff(log_arr, axis=0)])

        if self.from_weights:
            weights = arr
        else:
            row_sums = arr.sum(axis=1, keepdims=True)
            row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
            weights = arr / row_sums_safe

        gamma_star = np.full(T, np.nan)

        for t in range(self.min_periods, T):
            start = max(0, t - self.window)
            window_rets = rets[start:t]
            valid_mask = ~np.any(np.isnan(window_rets), axis=1)
            valid_rets = window_rets[valid_mask]

            if len(valid_rets) < self.min_periods:
                continue

            cov = np.cov(valid_rets, rowvar=False, ddof=1)
            if cov.ndim == 0:
                continue

            pi = weights[t]
            pi = np.clip(pi, 0, None)
            pi_sum = pi.sum()
            if pi_sum > 0:
                pi = pi / pi_sum
            else:
                continue

            weighted_var = float(np.dot(pi, np.diag(cov)))
            port_var = float(pi @ cov @ pi)
            gamma_star[t] = 0.5 * (weighted_var - port_var)

        return gamma_star.reshape(-1, 1)

    def fit_transform(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
        y: Any = None,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Fit and transform."""
        return self.fit(X, y).transform(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters."""
        return {
            "window": self.window,
            "from_weights": self.from_weights,
            "min_periods": self.min_periods,
        }

    def set_params(self, **params: Any) -> ExcessGrowthFeature:
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return (
            f"ExcessGrowthFeature(window={self.window}, "
            f"from_weights={self.from_weights}, "
            f"min_periods={self.min_periods})"
        )
