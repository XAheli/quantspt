"""Regime detection for Stochastic Portfolio Theory.

Market regime identification is critical for SPT: the diversity conditions
(FKK Eq. 4.5) required for relative arbitrage do not always hold. Regime
detection identifies WHEN conditions are favorable for FGP deployment.

Two complementary approaches:
  - HMMRegimeDetector: Hidden Markov Model for global regime states
    (e.g., diverse vs. concentrated markets)
  - ChangepointDetector: Online detection of structural breaks in
    diversity condition dynamics

References
----------
Fernholz, Karatzas & Kardaras, "Diversity and Relative Arbitrage in
Equity Markets," Finance and Stochastics 9:1-27, 2005 (FKK).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _require_hmmlearn() -> None:
    """Raise ImportError if hmmlearn is missing."""
    try:
        import hmmlearn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "HMMRegimeDetector requires hmmlearn. "
            "Install with: pip install quantspt[ml]"
        ) from e


def _require_ruptures() -> None:
    """Raise ImportError if ruptures is missing."""
    try:
        import ruptures  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ChangepointDetector requires ruptures. "
            "Install with: pip install quantspt[ml]"
        ) from e


class HMMRegimeDetector:
    """Hidden Markov Model for market regime detection.

    Detects regimes in market structure — typically two states
    corresponding to diverse (favorable for FGP) and concentrated
    (unfavorable) market conditions.

    The HMM models the observable features (diversity indices, growth
    rates, volatility) as emissions from hidden regime states, with
    Gaussian emission distributions and a Markov transition structure.

    Parameters
    ----------
    n_regimes : int
        Number of hidden states (regimes). Default 2.
    covariance_type : str
        Type of covariance for emissions: 'full', 'diag', 'spherical'.
        Default 'full'.
    n_iter : int
        Maximum EM iterations. Default 100.
    random_state : int | None
        Random seed for reproducibility.

    References
    ----------
    FKK Eq. 4.5 — diversity condition for relative arbitrage.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int | None = None,
    ) -> None:
        _require_hmmlearn()
        self._n_regimes = n_regimes
        self._covariance_type = covariance_type
        self._n_iter = n_iter
        self._random_state = random_state
        self._model: Any = None
        self._fitted = False

    @property
    def n_regimes(self) -> int:
        """Number of regimes."""
        return self._n_regimes

    @property
    def transition_matrix(self) -> NDArray[np.float64]:
        """Regime transition probability matrix P[i,j] = P(j|i).

        Returns
        -------
        ndarray of shape (n_regimes, n_regimes)
            Row-stochastic transition matrix.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")
        return np.array(self._model.transmat_, dtype=np.float64)

    def fit(
        self,
        features: NDArray[np.float64],
        *,
        n_regimes: int | None = None,
        **kwargs: Any,
    ) -> HMMRegimeDetector:
        """Fit the HMM to feature data.

        Parameters
        ----------
        features : ndarray of shape (T, d)
            Feature matrix. Typical features:
            - Market weight entropy (diversity measure)
            - Excess growth rate γ*_μ
            - Realized volatility
            - Herfindahl-Hirschman index
        n_regimes : int, optional
            Override number of regimes.
        **kwargs
            Additional hmmlearn parameters.

        Returns
        -------
        HMMRegimeDetector
            The fitted detector.
        """
        from hmmlearn.hmm import GaussianHMM

        if n_regimes is not None:
            self._n_regimes = n_regimes

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        self._model = GaussianHMM(
            n_components=self._n_regimes,
            covariance_type=self._covariance_type,
            n_iter=self._n_iter,
            random_state=self._random_state,
        )
        self._model.fit(features)
        self._fitted = True
        return self

    def predict(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Predict regime labels via Viterbi decoding.

        Parameters
        ----------
        features : ndarray of shape (T, d)
            Feature matrix.

        Returns
        -------
        ndarray of shape (T,)
            Integer regime labels in [0, n_regimes).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        return np.asarray(self._model.predict(features), dtype=np.int64)

    def predict_proba(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict regime posterior probabilities.

        Parameters
        ----------
        features : ndarray of shape (T, d)
            Feature matrix.

        Returns
        -------
        ndarray of shape (T, n_regimes)
            Posterior probability of each regime at each time step.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        return np.asarray(self._model.predict_proba(features), dtype=np.float64)

    def forecast_diversity(self, horizon: int) -> NDArray[np.float64]:
        """Forecast regime probabilities over a future horizon.

        Uses the transition matrix to compute multi-step regime
        probabilities from the current stationary distribution.

        Parameters
        ----------
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        ndarray of shape (horizon, n_regimes)
            Predicted regime probabilities at each future step.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")
        P = self.transition_matrix
        stationary = self._stationary_distribution(P)
        forecasts = np.zeros((horizon, self._n_regimes))
        current = stationary
        for t in range(horizon):
            current = current @ P
            forecasts[t] = current
        return forecasts

    @staticmethod
    def _stationary_distribution(P: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute stationary distribution of transition matrix."""
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return stationary.astype(np.float64)


class ChangepointDetector:
    """Online changepoint detection for diversity condition shifts.

    Identifies structural breaks in time series of diversity measures,
    signaling when the market transitions between regimes where FGP
    strategies are viable vs. not.

    Uses the PELT (Pruned Exact Linear Time) algorithm for efficient
    offline changepoint detection, or sliding-window methods for
    online detection.

    Parameters
    ----------
    model : str
        Cost model: 'l2' (mean shift), 'rbf' (kernel), 'normal'
        (mean and variance). Default 'l2'.
    penalty : float
        Penalty for adding a changepoint. Higher values produce
        fewer changepoints. Default 3.0.
    min_size : int
        Minimum segment length between changepoints. Default 20.

    References
    ----------
    FKK Eq. 4.5 — diversity condition for relative arbitrage.
    """

    def __init__(
        self,
        model: str = "l2",
        penalty: float = 3.0,
        min_size: int = 20,
    ) -> None:
        _require_ruptures()
        self._model_name = model
        self._penalty = penalty
        self._min_size = min_size
        self._changepoints: list[int] = []
        self._fitted = False

    @property
    def changepoints(self) -> list[int]:
        """Detected changepoint indices (sorted)."""
        return self._changepoints

    def fit(
        self,
        signal: NDArray[np.float64],
        *,
        penalty: float | None = None,
        **kwargs: Any,
    ) -> ChangepointDetector:
        """Detect changepoints in the signal.

        Parameters
        ----------
        signal : ndarray of shape (T,) or (T, d)
            Time series to analyze (e.g., diversity index over time).
        penalty : float, optional
            Override penalty value.
        **kwargs
            Additional ruptures parameters.

        Returns
        -------
        ChangepointDetector
            The fitted detector.
        """
        import ruptures

        if penalty is not None:
            self._penalty = penalty

        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        algo = ruptures.Pelt(
            model=self._model_name,
            min_size=self._min_size,
        ).fit(signal)

        breakpoints = algo.predict(pen=self._penalty)
        self._changepoints = [bp for bp in breakpoints if bp < len(signal)]
        self._fitted = True
        return self

    def predict(
        self,
        signal: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Assign regime labels based on detected changepoints.

        Segments between changepoints get consecutive integer labels.

        Parameters
        ----------
        signal : ndarray of shape (T,) or (T, d)
            Time series.

        Returns
        -------
        ndarray of shape (T,)
            Regime labels (0, 1, 2, ... for each segment).
        """
        if signal.ndim == 1:
            T = len(signal)
        else:
            T = signal.shape[0]

        if not self._fitted:
            self.fit(signal)

        labels = np.zeros(T, dtype=np.int64)
        boundaries = [0, *self._changepoints, T]
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            labels[start:end] = i

        return labels


__all__ = [
    "ChangepointDetector",
    "HMMRegimeDetector",
]
