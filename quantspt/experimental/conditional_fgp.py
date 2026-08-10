"""Covariance-Conditional Functionally Generated Portfolios.

Since drift g(t) = -(1/2G) sum D^2 G mu_i mu_j tau^mu_ij and the
relative covariance structure tau^mu explains 99.6% of drift variation,
the optimal generating function should adapt its shape based on the
current covariance regime.

Instead of learning an entire neural generating function (which collapses
on real data), this module learns a 1-parameter decision: which diversity
parameter p to use, conditioned on measurable covariance features.

Architecture:
  1. Extract low-dimensional features from the current covariance matrix:
     - Average pairwise correlation
     - Max eigenvalue ratio (PCA concentration)
     - Effective rank (eigenvalue entropy)
     - Eigenvalue Herfindahl index
  2. Map features to optimal p in (0.05, 0.95)
  3. Use DiversityGenerator(p) for actual weight computation

Key properties:
  - 1 output parameter vs thousands of network weights
  - Interpretable (feature importances show what drives the decision)
  - Cannot collapse (p is bounded, DiversityGenerator always works)
  - Leverages the causal mechanism: covariance regime -> drift
  - Includes boundary-robustness regularization against concentration
    shocks, since losses are 100% boundary-driven
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.generating_functions import (
    DiversityGenerator,
    GeneratingFunction,
)
from ..ml._protocols import LearnedGeneratingFunction

_log = logging.getLogger(__name__)

__all__ = [
    "BoundaryRobustnessRegularizer",
    "ConditionalFGPConfig",
    "CovarianceConditionalFGP",
    "CovarianceFeatureExtractor",
    "cost_optimal_p",
    "optimal_p_for_cost_level",
]


DEFAULT_FEATURES = [
    "avg_correlation",
    "max_eigenvalue_ratio",
    "effective_rank",
    "eigenvalue_herfindahl",
]

P_MIN = 0.05
P_MAX = 0.95

_VALID_FEATURES = frozenset(
    {
        "avg_correlation",
        "max_eigenvalue_ratio",
        "effective_rank",
        "eigenvalue_herfindahl",
        "trace_normalized",
        "condition_number",
    }
)


# ---------------------------------------------------------------------------
# Covariance feature extraction
# ---------------------------------------------------------------------------


class CovarianceFeatureExtractor:
    """Extract low-dimensional summary features from a covariance matrix.

    These features capture the covariance regime that drives 99.6% of
    drift variation in the diversity generator.

    Parameters
    ----------
    features : list of str, optional
        Which features to extract. Defaults to
        ``["avg_correlation", "max_eigenvalue_ratio",
          "effective_rank", "eigenvalue_herfindahl"]``.
    """

    def __init__(self, features: list[str] | None = None) -> None:
        self._features = features or DEFAULT_FEATURES[:]
        for f in self._features:
            if f not in _VALID_FEATURES:
                raise ValueError(
                    f"Unknown feature {f!r}. Valid: {sorted(_VALID_FEATURES)}"
                )

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names."""
        return self._features[:]

    @property
    def n_features(self) -> int:
        """Number of features extracted per covariance matrix."""
        return len(self._features)

    def extract(self, cov_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract feature vector from a single covariance matrix.

        Parameters
        ----------
        cov_matrix : ndarray of shape (n, n)
            Symmetric positive (semi-)definite covariance matrix.

        Returns
        -------
        ndarray of shape (n_features,)
        """
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError(f"Expected square matrix, got shape {cov_matrix.shape}")

        n = cov_matrix.shape[0]
        eigenvalues: NDArray[np.float64] = np.asarray(
            np.linalg.eigvalsh(cov_matrix), dtype=np.float64
        )
        eigenvalues = np.maximum(eigenvalues, 0.0)

        diag = np.diag(cov_matrix)
        std = np.sqrt(np.maximum(diag, 1e-30))
        corr = cov_matrix / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        dispatch = {
            "avg_correlation": lambda: _avg_correlation(corr, n),
            "max_eigenvalue_ratio": lambda: _max_eigenvalue_ratio(eigenvalues),
            "effective_rank": lambda: _effective_rank(eigenvalues),
            "eigenvalue_herfindahl": lambda: _eigenvalue_herfindahl(eigenvalues),
            "trace_normalized": lambda: _trace_normalized(eigenvalues, n),
            "condition_number": lambda: _condition_number(eigenvalues),
        }

        result = np.zeros(len(self._features))
        for i, name in enumerate(self._features):
            result[i] = dispatch[name]()  # type: ignore[no-untyped-call]
        return result

    def extract_batch(
        self, cov_matrices: list[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """Extract features from multiple covariance matrices.

        Returns
        -------
        ndarray of shape (len(cov_matrices), n_features)
        """
        return np.array([self.extract(c) for c in cov_matrices])


def _avg_correlation(corr: NDArray[np.float64], n: int) -> float:
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(corr[mask]))


def _max_eigenvalue_ratio(eigenvalues: NDArray[np.float64]) -> float:
    total = float(np.sum(eigenvalues))
    if total <= 0:
        return 1.0
    return float(np.max(eigenvalues)) / total


def _effective_rank(eigenvalues: NDArray[np.float64]) -> float:
    """exp(entropy of normalized eigenvalue distribution)."""
    total = float(np.sum(eigenvalues))
    if total <= 0:
        return 1.0
    probs = eigenvalues / total
    probs = probs[probs > 1e-30]
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(np.exp(entropy))


def _eigenvalue_herfindahl(eigenvalues: NDArray[np.float64]) -> float:
    total = float(np.sum(eigenvalues))
    if total <= 0:
        return 1.0
    shares = eigenvalues / total
    return float(np.sum(shares**2))


def _trace_normalized(eigenvalues: NDArray[np.float64], n: int) -> float:
    if n == 0:
        return 0.0
    return float(np.sum(eigenvalues)) / n


def _condition_number(eigenvalues: NDArray[np.float64]) -> float:
    pos = eigenvalues[eigenvalues > 1e-30]
    if len(pos) == 0:
        return 1.0
    return float(np.max(pos) / np.min(pos))


# ---------------------------------------------------------------------------
# Boundary-robustness regularization
# ---------------------------------------------------------------------------


class BoundaryRobustnessRegularizer:
    """Penalize generating functions vulnerable to concentration spikes.

    Since strategy losses are 100% boundary-driven (market concentration
    increases), this regularizer stress-tests a generating function against
    concentration shock scenarios and penalizes sensitivity.

    Parameters
    ----------
    shock_magnitude : float
        Fractional increase applied to a top stock's weight. Must be in (0, 1).
    n_scenarios : int
        Number of concentration shock scenarios to evaluate.
    penalty_weight : float
        Multiplier for the mean squared boundary penalty.
    """

    def __init__(
        self,
        shock_magnitude: float = 0.5,
        n_scenarios: int = 5,
        penalty_weight: float = 0.1,
    ) -> None:
        if shock_magnitude <= 0 or shock_magnitude >= 1:
            raise ValueError(
                f"shock_magnitude must be in (0, 1), got {shock_magnitude}"
            )
        if n_scenarios < 1:
            raise ValueError(f"n_scenarios must be >= 1, got {n_scenarios}")
        self._shock = shock_magnitude
        self._n_scenarios = n_scenarios
        self._weight = penalty_weight

    @property
    def penalty_weight(self) -> float:
        """Multiplier applied to the raw penalty."""
        return self._weight

    def concentration_scenarios(
        self, mu: NDArray[np.float64]
    ) -> list[NDArray[np.float64]]:
        """Generate concentration shock scenarios.

        Each scenario boosts one top stock's weight by ``shock_magnitude``
        and redistributes from the rest, staying on the simplex.

        Parameters
        ----------
        mu : ndarray of shape (n,)
            Current market weights.

        Returns
        -------
        list of ndarray, each shape (n,)
            Stressed market-weight vectors.
        """
        n = len(mu)
        ranked = np.argsort(mu)[::-1]
        scenarios: list[NDArray[np.float64]] = []

        for k in range(min(self._n_scenarios, n)):
            mu_stressed = mu.copy()
            top_idx = ranked[k]
            boost = mu_stressed[top_idx] * self._shock
            mu_stressed[top_idx] += boost

            others = np.arange(n) != top_idx
            other_sum = float(np.sum(mu_stressed[others]))
            if other_sum > boost:
                mu_stressed[others] -= mu_stressed[others] * (boost / other_sum)
            else:
                mu_stressed[others] = 1e-10

            mu_stressed = np.maximum(mu_stressed, 1e-10)
            mu_stressed /= mu_stressed.sum()
            scenarios.append(mu_stressed)

        return scenarios

    def penalty(self, G: GeneratingFunction, mu: NDArray[np.float64]) -> float:
        """Compute boundary-robustness penalty.

        Measures how much ``log(G(mu_stressed) / G(mu))`` can go
        negative when market concentration increases.

        Returns
        -------
        float
            Non-negative penalty. Zero means fully robust.
        """
        G_current = G(mu)
        if G_current <= 0:
            return float("inf")

        log_G_current = np.log(G_current)
        penalties: list[float] = []

        for mu_stressed in self.concentration_scenarios(mu):
            G_stressed = G(mu_stressed)
            if G_stressed <= 0:
                penalties.append(1.0)
                continue
            boundary = np.log(G_stressed) - log_G_current
            if boundary < 0:
                penalties.append(boundary**2)

        if not penalties:
            return 0.0
        return self._weight * float(np.mean(penalties))

    def select_robust_p(
        self,
        mu: NDArray[np.float64],
        p_candidates: NDArray[np.float64] | None = None,
    ) -> tuple[float, NDArray[np.float64]]:
        """Find the p value most robust to concentration shocks.

        Parameters
        ----------
        mu : ndarray of shape (n,)
            Current market weights.
        p_candidates : ndarray, optional
            Grid of p values. Defaults to 50 points in [0.05, 0.95].

        Returns
        -------
        best_p : float
        penalties : ndarray
            Boundary penalty at each candidate p.
        """
        if p_candidates is None:
            p_candidates = np.asarray(np.linspace(P_MIN, P_MAX, 50), dtype=np.float64)

        penalties: NDArray[np.float64] = np.array(
            [self.penalty(DiversityGenerator(float(p)), mu) for p in p_candidates],
            dtype=np.float64,
        )

        best_idx = int(np.argmin(penalties))
        return float(p_candidates[best_idx]), penalties


# ---------------------------------------------------------------------------
# Cost-aware p selection
# ---------------------------------------------------------------------------


def cost_optimal_p(
    cost_bps: float,
    rebalancing_days: int = 21,
) -> float:
    """Select the diversity parameter p based on transaction cost level.

    Empirical relationship from systematic analysis of p-sweep across
    cost levels with proper market-cap weighting::

        optimal_p ~ 0.09 + 45 * cost_fraction

    where ``cost_fraction = cost_bps / 10000``.  Equivalently,
    ``optimal_p ~ 0.09 + 0.0045 * cost_bps``.

    Calibrated to: 0bps -> p~0.09, 10bps -> p~0.14, 50bps -> p~0.31,
    100bps -> p~0.54.

    Lower costs -> lower p (more aggressive small-cap tilt).
    Higher costs -> higher p (less turnover).

    Parameters
    ----------
    cost_bps : float
        Transaction cost in basis points.
    rebalancing_days : int
        Days between rebalances (affects optimal aggression).

    Returns
    -------
    float
        Optimal p in [0.05, 0.95].
    """
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be non-negative, got {cost_bps}")

    cost_fraction = cost_bps / 10_000.0
    p = 0.09 + 45.0 * cost_fraction
    freq_factor = 21.0 / max(rebalancing_days, 1)
    p = 0.09 + (p - 0.09) * freq_factor

    return float(np.clip(p, P_MIN, P_MAX))


def optimal_p_for_cost_level(
    returns: NDArray[np.float64],
    market_weights: NDArray[np.float64],
    cost_bps: float,
    *,
    p_grid: NDArray[np.float64] | None = None,
    rebalancing_days: int = 21,
) -> dict[str, Any]:
    """Find the empirically optimal p for a given cost level via grid search.

    Runs a backtest at each candidate p, computing net-of-cost excess
    returns, and returns the best.

    Parameters
    ----------
    returns : ndarray of shape (T, n)
        Asset return series (1 + r per period).
    market_weights : ndarray of shape (T+1, n)
        Market weights at each time step.
    cost_bps : float
        Transaction cost in basis points.
    p_grid : ndarray, optional
        Grid of p values. Defaults to 20 points in [0.05, 0.95].
    rebalancing_days : int
        Days between rebalances.

    Returns
    -------
    dict
        ``optimal_p``, ``optimal_net_excess``, ``p_values``,
        ``net_excess_returns``, ``formula_p``.
    """
    from ..backtesting.engine import BacktestConfig, BacktestEngine
    from ..backtesting.execution import ProportionalCostExecution
    from ..backtesting.rebalancing import CalendarRebalancer, Frequency

    if p_grid is None:
        p_grid = np.asarray(np.linspace(P_MIN, P_MAX, 20), dtype=np.float64)

    freq_map = {
        1: Frequency.DAILY,
        5: Frequency.WEEKLY,
        21: Frequency.MONTHLY,
        63: Frequency.QUARTERLY,
    }
    freq = freq_map.get(rebalancing_days, Frequency.MONTHLY)

    net_excess: list[float] = []
    n_years = len(returns) / 252.0

    init_w = market_weights[0].copy()
    init_w = np.maximum(init_w, 1e-10)
    init_w /= init_w.sum()

    for p_val in p_grid:
        gen = DiversityGenerator(float(p_val))
        engine = BacktestEngine(
            weight_func=gen.weights,
            returns=returns,
            initial_weights=init_w,
            rebalancer=CalendarRebalancer(freq),
            execution=ProportionalCostExecution(cost_bps=cost_bps),
            config=BacktestConfig(initial_value=1.0),
        )
        result = engine.run().data
        ann_excess = result.log_relative_return() / max(n_years, 1e-6)
        net_excess.append(ann_excess)

    net_arr = np.array(net_excess)
    best_idx = int(np.argmax(net_arr))

    return {
        "optimal_p": float(p_grid[best_idx]),
        "optimal_net_excess": float(net_arr[best_idx]),
        "p_values": p_grid,
        "net_excess_returns": net_arr,
        "formula_p": cost_optimal_p(cost_bps, rebalancing_days),
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConditionalFGPConfig:
    """Configuration for CovarianceConditionalFGP.

    Parameters
    ----------
    features : list of str
        Covariance features to extract.
    p_min : float
        Minimum diversity parameter.
    p_max : float
        Maximum diversity parameter.
    cov_window : int
        Rolling window (trading days) for covariance estimation.
    rebalancing_days : int
        Days between rebalances.
    cost_bps : float
        Transaction cost in basis points.
    p_grid_size : int
        Number of p values to sweep per training window.
    boundary_penalty_weight : float
        Weight for boundary-robustness penalty. 0 disables it.
    fallback_p : float
        Default p before the model is fitted.
    n_estimators : int
        Number of gradient boosting rounds for the feature->p model.
    """

    features: list[str] = field(default_factory=lambda: DEFAULT_FEATURES[:])
    p_min: float = P_MIN
    p_max: float = P_MAX
    cov_window: int = 63
    rebalancing_days: int = 21
    cost_bps: float = 10.0
    p_grid_size: int = 20
    boundary_penalty_weight: float = 0.1
    fallback_p: float = 0.3
    n_estimators: int = 50


# ---------------------------------------------------------------------------
# CovarianceConditionalFGP — the main class
# ---------------------------------------------------------------------------


class CovarianceConditionalFGP:
    """Generating function that adapts to covariance regimes.

    Since ``g(t) = -(1/2G) sum D^2 G mu_i mu_j tau^mu_ij`` and the
    relative covariance structure tau^mu explains 99.6% of drift
    variation, the optimal diversity parameter p should depend on the
    current covariance state.

    This class learns ``p(Sigma)`` — the optimal diversity parameter as
    a function of low-dimensional covariance features. This is a
    1-parameter adaptation (not thousands of network weights), making it
    learnable from limited financial data.

    Parameters
    ----------
    config : ConditionalFGPConfig, optional
        Configuration. Uses sensible defaults if omitted.

    Examples
    --------
    ::

        model = CovarianceConditionalFGP()
        model.fit(market_weights, covariance_matrices, returns, cost_bps=10)

        # Regime-adaptive weights
        pi = model.weights(current_mu, current_cov)

        # Convert to GeneratingFunction for master formula, etc.
        G = model.to_generating_function()
    """

    def __init__(self, config: ConditionalFGPConfig | None = None) -> None:
        self._config = config or ConditionalFGPConfig()
        self._extractor = CovarianceFeatureExtractor(self._config.features)
        self._boundary_reg: BoundaryRobustnessRegularizer | None = (
            BoundaryRobustnessRegularizer(
                penalty_weight=self._config.boundary_penalty_weight,
            )
            if self._config.boundary_penalty_weight > 0
            else None
        )
        self._p_model: Any = None
        self._fitted = False
        self._training_metadata: dict[str, Any] = {}
        self._n_assets: int = 0
        self._feature_importances: Any = None
        self._fallback_p = self._config.fallback_p

    @property
    def config(self) -> ConditionalFGPConfig:
        """Current configuration."""
        return self._config

    @property
    def fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._fitted

    @property
    def feature_importances(self) -> Any:
        """Feature importances from the fitted model (tree-based)."""
        return self._feature_importances

    @property
    def training_metadata(self) -> dict[str, Any]:
        """Diagnostics from the last ``fit()`` call."""
        return self._training_metadata

    @property
    def n_assets(self) -> int:
        """Number of assets the model was fitted on."""
        return self._n_assets

    @property
    def fallback_p(self) -> float:
        """Current fallback diversity parameter."""
        return self._fallback_p

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_covariance_features(
        self, cov_matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Extract covariance-regime features from a covariance matrix.

        Parameters
        ----------
        cov_matrix : ndarray of shape (n, n)

        Returns
        -------
        ndarray of shape (n_features,)
        """
        return self._extractor.extract(cov_matrix)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def optimal_p(self, cov_features: NDArray[np.float64]) -> float:
        """Predict optimal p from covariance features.

        Falls back to the cost-based formula if the model is not fitted.
        """
        if not self._fitted or self._p_model is None:
            return cost_optimal_p(self._config.cost_bps, self._config.rebalancing_days)

        features_2d = cov_features.reshape(1, -1)
        p_raw = float(self._p_model.predict(features_2d)[0])
        return float(np.clip(p_raw, self._config.p_min, self._config.p_max))

    def weights(
        self,
        mu: NDArray[np.float64],
        cov_matrix: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Generate portfolio weights conditional on covariance regime.

        Parameters
        ----------
        mu : ndarray of shape (n,)
            Current market weights.
        cov_matrix : ndarray of shape (n, n), optional
            Current covariance matrix. Uses fallback p when absent.

        Returns
        -------
        ndarray of shape (n,)
            Portfolio weights.
        """
        if cov_matrix is not None:
            features = self.extract_covariance_features(cov_matrix)
            p = self.optimal_p(features)
        else:
            p = self._fallback_p

        return DiversityGenerator(p).weights(mu)

    def weights_from_mu_only(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Weight function using fallback p (no covariance input).

        Convenience for the backtesting engine, which expects mu -> weights.
        """
        return DiversityGenerator(self._fallback_p).weights(mu)

    def make_weight_func(
        self,
        cov_matrix: NDArray[np.float64],
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Return a mu->weights callable bound to a specific covariance.

        Useful for the backtesting engine.
        """
        features = self.extract_covariance_features(cov_matrix)
        p = self.optimal_p(features)
        gen = DiversityGenerator(p)
        return gen.weights

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        market_weights: NDArray[np.float64],
        covariance_matrices: list[NDArray[np.float64]],
        returns: NDArray[np.float64],
        *,
        cost_bps: float | None = None,
        validation_split: float = 0.2,
        **kwargs: Any,
    ) -> CovarianceConditionalFGP:
        """Learn the covariance-feature -> optimal-p mapping.

        For each historical window:
          1. Extract covariance features
          2. Sweep p values to find which p was optimal (net of costs)
          3. Build training pairs ``(features, optimal_p)``
          4. Fit a gradient-boosting model

        Parameters
        ----------
        market_weights : ndarray of shape (T, n)
            Market weights time series.
        covariance_matrices : list of ndarray, each (n, n)
            Covariance matrix at each time step. Length must match T.
        returns : ndarray of shape (T-1, n) or (T, n)
            Asset return series ``(1 + r)`` per period.
        cost_bps : float, optional
            Override cost level. Defaults to config value.
        validation_split : float
            Fraction of windows held out for validation.

        Returns
        -------
        self
        """
        from sklearn.ensemble import GradientBoostingRegressor

        if cost_bps is None:
            cost_bps = self._config.cost_bps

        T, n = market_weights.shape
        self._n_assets = n

        _log.info(
            "Fitting CovarianceConditionalFGP: T=%d, n=%d, cost=%dbps",
            T,
            n,
            int(cost_bps),
        )

        window = self._config.rebalancing_days * 3
        p_grid: NDArray[np.float64] = np.asarray(
            np.linspace(
                self._config.p_min, self._config.p_max, self._config.p_grid_size
            ),
            dtype=np.float64,
        )
        step = max(self._config.rebalancing_days, 1)

        X_train: list[NDArray[np.float64]] = []
        y_train: list[float] = []

        for t in range(0, T - window, step):
            if t >= len(covariance_matrices):
                break

            cov_t = covariance_matrices[t]
            if cov_t is None or np.any(np.isnan(cov_t)):
                continue

            features = self._extractor.extract(cov_t)

            end = min(t + window, len(returns))
            if end - t < self._config.rebalancing_days:
                continue

            window_returns = returns[t:end]
            w_end = min(t + window + 1, T)
            window_weights = market_weights[t:w_end]

            if len(window_returns) < self._config.rebalancing_days:
                continue
            if len(window_weights) < 2:
                continue

            best_p = self._find_best_p_for_window(
                window_returns, window_weights, p_grid, cost_bps
            )

            if self._boundary_reg is not None:
                mu_t = market_weights[t]
                _, penalties = self._boundary_reg.select_robust_p(
                    mu_t, np.asarray(p_grid, dtype=np.float64)
                )
                robust_idx = int(np.argmin(penalties))
                robust_p = float(p_grid[robust_idx])
                alpha = min(self._config.boundary_penalty_weight, 0.5)
                best_p = (1 - alpha) * best_p + alpha * robust_p
                best_p = float(np.clip(best_p, self._config.p_min, self._config.p_max))

            X_train.append(features)
            y_train.append(best_p)

        if len(X_train) < 5:
            _log.warning(
                "Insufficient training samples (%d). Using formula-based fallback.",
                len(X_train),
            )
            self._fallback_p = cost_optimal_p(cost_bps, self._config.rebalancing_days)
            self._fitted = True
            self._training_metadata = {
                "n_samples": len(X_train),
                "status": "fallback_to_formula",
                "fallback_p": self._fallback_p,
            }
            return self

        X = np.array(X_train)
        y = np.array(y_train)

        split_idx = int(len(X) * (1 - validation_split))
        X_fit, X_val = X[:split_idx], X[split_idx:]
        y_fit, y_val = y[:split_idx], y[split_idx:]

        if len(X_fit) < 3:
            X_fit, y_fit = X, y
            X_val, y_val = X, y

        self._p_model = GradientBoostingRegressor(
            n_estimators=self._config.n_estimators,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._p_model.fit(X_fit, y_fit)

        self._feature_importances = np.array(self._p_model.feature_importances_)

        y_pred_val = self._p_model.predict(X_val)
        val_mae = float(np.mean(np.abs(y_val - y_pred_val)))

        var_y = float(np.var(y_val))
        if len(y_val) > 1 and var_y > 1e-12:
            ss_res = float(np.sum((y_val - y_pred_val) ** 2))
            ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
            val_r2 = 1 - ss_res / ss_tot
        else:
            val_r2 = 0.0

        self._fallback_p = float(np.median(self._p_model.predict(X)))

        self._fitted = True
        self._training_metadata = {
            "n_samples": len(X),
            "n_train": len(X_fit),
            "n_val": len(X_val),
            "val_mae": val_mae,
            "val_r2": val_r2,
            "mean_optimal_p": float(np.mean(y)),
            "std_optimal_p": float(np.std(y)),
            "feature_importances": dict(
                zip(
                    self._config.features,
                    self._feature_importances.tolist(),
                    strict=False,
                )
            ),
            "fallback_p": self._fallback_p,
            "status": "fitted",
        }

        _log.info(
            "Fitted: n_samples=%d, val_MAE=%.4f, val_R2=%.4f, mean_p=%.3f +/- %.3f",
            len(X),
            val_mae,
            val_r2,
            float(np.mean(y)),
            float(np.std(y)),
        )
        return self

    def _find_best_p_for_window(
        self,
        window_returns: NDArray[np.float64],
        window_weights: NDArray[np.float64],
        p_grid: NDArray[np.float64],
        cost_bps: float,
    ) -> float:
        """Find the p maximising net excess return in a local window."""
        from ..backtesting.engine import BacktestConfig, BacktestEngine
        from ..backtesting.execution import ProportionalCostExecution
        from ..backtesting.rebalancing import CalendarRebalancer, Frequency

        best_p = self._config.fallback_p
        best_return = -np.inf

        init_w = window_weights[0].copy()
        init_w = np.maximum(init_w, 1e-10)
        init_w /= init_w.sum()

        for p_val in p_grid:
            gen = DiversityGenerator(float(p_val))
            try:
                engine = BacktestEngine(
                    weight_func=gen.weights,
                    returns=window_returns,
                    initial_weights=init_w,
                    rebalancer=CalendarRebalancer(Frequency.MONTHLY),
                    execution=ProportionalCostExecution(cost_bps=cost_bps),
                    config=BacktestConfig(initial_value=1.0),
                )
                result = engine.run().data
                excess = result.log_relative_return()
                if excess > best_return:
                    best_return = excess
                    best_p = float(p_val)
            except (ValueError, RuntimeError, ArithmeticError):
                continue

        return best_p

    # ------------------------------------------------------------------
    # GeneratingFunctionModel protocol
    # ------------------------------------------------------------------

    def generating_function(self, mu: NDArray[np.float64]) -> float:
        """Evaluate G(mu) using the current fallback p."""
        return DiversityGenerator(self._fallback_p)(mu)

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gradient of log G(mu) using the current fallback p."""
        return DiversityGenerator(self._fallback_p).log_gradient(mu)

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Hessian D^2 G(mu) using the current fallback p."""
        return DiversityGenerator(self._fallback_p).hessian(mu)

    def to_generating_function(self) -> GeneratingFunction:
        """Convert to core GeneratingFunction for master formula, etc."""
        return LearnedGeneratingFunction(
            self,  # type: ignore[arg-type]
            name_str=f"ConditionalFGP(p~{self._fallback_p:.3f})",
            n_assets=max(self._n_assets, 5),
            skip_validation=True,
        )
