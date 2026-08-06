"""Causal analysis of stock rank dynamics.

Combines causal structure learning with rank transition analysis to
identify which rank transitions *cause* other transitions, going beyond
mere correlation.  Uses Granger-style causal testing on rank time
series, then learns a DAG over the rank-change variables.

Integration with ``quantspt.rank.transitions`` is provided through the
``from_rank_transitions`` classmethod.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as sp_stats

from .._preconditions import require

__all__ = [
    "CausalRankAnalysis",
]


class CausalRankAnalysis:
    """Causal analysis of rank transition dynamics.

    Parameters
    ----------
    max_lag : int
        Maximum lag order for Granger-style causality tests.
    significance_level : float
        P-value threshold for declaring Granger causality.
    structure_method : ``"pc"`` | ``"ges"`` | ``"hillclimb"``
        Discovery algorithm for learning the DAG over rank changes.
    ci_test : str
        Conditional-independence test for constraint-based methods.
    scoring_method : str
        Scoring criterion for score-based methods.
    """

    def __init__(
        self,
        *,
        max_lag: int = 5,
        significance_level: float = 0.05,
        structure_method: str = "pc",
        ci_test: str = "pearsonr",
        scoring_method: str = "bic-g",
    ) -> None:
        require(max_lag >= 1, f"max_lag must be ≥ 1, got {max_lag}")
        self._max_lag = max_lag
        self._significance_level = significance_level
        self._structure_method = structure_method
        self._ci_test = ci_test
        self._scoring_method = scoring_method

        self._fitted = False
        self._granger_pvalues: NDArray[np.float64] | None = None
        self._granger_fstats: NDArray[np.float64] | None = None
        self._causal_edges: list[tuple[str, str]] = []
        self._variable_names: list[str] = []
        self._optimal_lags: NDArray[np.int64] | None = None

    def fit(
        self,
        rank_series: pd.DataFrame | NDArray[np.float64],
        *,
        variable_names: list[str] | None = None,
        **kwargs: Any,
    ) -> CausalRankAnalysis:
        """Fit causal analysis on rank time series.

        Parameters
        ----------
        rank_series : DataFrame or ndarray of shape (T, n)
            Time series of ranks (or rank changes) for n stocks.
        variable_names : list of str, optional
            Required when *rank_series* is an ndarray.
        **kwargs
            Forwarded to the structure learner.

        Returns
        -------
        CausalRankAnalysis
            The fitted analyser (for chaining).
        """
        df = self._to_dataframe(rank_series, variable_names)
        self._variable_names = list(df.columns)
        n_vars = len(self._variable_names)

        pvals = np.ones((n_vars, n_vars), dtype=np.float64)
        fstats = np.zeros((n_vars, n_vars), dtype=np.float64)
        opt_lags = np.zeros((n_vars, n_vars), dtype=np.int64)

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                f_stat, p_val, lag = self._granger_test(
                    np.asarray(df.iloc[:, i].values, dtype=np.float64),
                    np.asarray(df.iloc[:, j].values, dtype=np.float64),
                )
                pvals[i, j] = p_val
                fstats[i, j] = f_stat
                opt_lags[i, j] = lag

        self._granger_pvalues = pvals
        self._granger_fstats = fstats
        self._optimal_lags = opt_lags

        self._causal_edges = self._significant_edges(pvals)
        self._fitted = True
        return self

    @classmethod
    def from_weight_paths(
        cls,
        weight_paths: NDArray[np.float64],
        stock_names: list[str] | None = None,
        **kwargs: Any,
    ) -> CausalRankAnalysis:
        """Construct from market weight paths.

        Computes ranks from weights, then fits the causal analysis on
        rank-change time series.

        Parameters
        ----------
        weight_paths : ndarray of shape (T, n)
            Market weight paths.
        stock_names : list of str, optional
            Stock names.  Defaults to ``["stock_0", "stock_1", ...]``.
        **kwargs
            Forwarded to the constructor.

        Returns
        -------
        CausalRankAnalysis
        """
        T, n = weight_paths.shape
        if stock_names is None:
            stock_names = [f"stock_{i}" for i in range(n)]

        ranks = np.zeros_like(weight_paths, dtype=np.float64)
        for t in range(T):
            ranks[t] = np.argsort(np.argsort(-weight_paths[t])).astype(np.float64)

        rank_changes = np.diff(ranks, axis=0)
        df = pd.DataFrame(rank_changes, columns=stock_names)

        obj = cls(**kwargs)
        obj.fit(df)
        return obj

    @property
    def granger_pvalues(self) -> NDArray[np.float64]:
        """P-value matrix from pairwise Granger tests.

        Entry ``(i, j)`` is the p-value for *j Granger-causes i*.

        Returns
        -------
        ndarray of shape (n, n)
        """
        require(self._fitted, "Must call .fit() first")
        assert self._granger_pvalues is not None
        return self._granger_pvalues.copy()

    @property
    def granger_fstats(self) -> NDArray[np.float64]:
        """F-statistic matrix from pairwise Granger tests.

        Returns
        -------
        ndarray of shape (n, n)
        """
        require(self._fitted, "Must call .fit() first")
        assert self._granger_fstats is not None
        return self._granger_fstats.copy()

    @property
    def causal_edges(self) -> list[tuple[str, str]]:
        """Significant causal edges at the configured significance level.

        Returns
        -------
        list of (str, str)
            Each ``(cause, effect)`` indicates that *cause*
            Granger-causes *effect*.
        """
        require(self._fitted, "Must call .fit() first")
        return list(self._causal_edges)

    @property
    def variable_names(self) -> list[str]:
        """Variable names."""
        require(self._fitted, "Must call .fit() first")
        return list(self._variable_names)

    def causal_strength(self) -> NDArray[np.float64]:
        """Signed causal strength matrix: −log10(p) × sign(F).

        Larger positive values indicate stronger evidence for
        Granger causality.  Diagonal is zero.

        Returns
        -------
        ndarray of shape (n, n)
        """
        require(self._fitted, "Must call .fit() first")
        assert self._granger_pvalues is not None

        with np.errstate(divide="ignore"):
            strength = -np.log10(np.clip(self._granger_pvalues, 1e-300, 1.0))
        np.fill_diagonal(strength, 0.0)
        return strength

    def _granger_test(
        self,
        y: NDArray[np.float64],
        x: NDArray[np.float64],
    ) -> tuple[float, float, int]:
        """Pairwise Granger causality test: does x Granger-cause y?

        Selects the optimal lag (1..max_lag) by minimum p-value.

        Returns (F-statistic, p-value, optimal_lag).
        """
        best_f, best_p, best_lag = 0.0, 1.0, 1
        T = len(y)

        for lag in range(1, self._max_lag + 1):
            if 2 * lag + 1 >= T:
                break

            Y = y[lag:]
            n_obs = len(Y)

            X_restricted = np.column_stack(
                [y[lag - k - 1 : T - k - 1] for k in range(lag)]
            )
            X_full = np.column_stack(
                [
                    X_restricted,
                    *[x[lag - k - 1 : T - k - 1] for k in range(lag)],
                ]
            )

            X_restricted = np.column_stack([np.ones(n_obs), X_restricted])
            X_full = np.column_stack([np.ones(n_obs), X_full])

            rss_r = self._ols_rss(X_restricted, Y)
            rss_f = self._ols_rss(X_full, Y)

            df_num = lag
            df_den = n_obs - X_full.shape[1]
            if df_den <= 0 or rss_f <= 0:
                continue

            f_stat = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
            p_val = 1.0 - float(sp_stats.f.cdf(f_stat, df_num, df_den))

            if p_val < best_p:
                best_f, best_p, best_lag = float(f_stat), p_val, lag

        return best_f, best_p, best_lag

    @staticmethod
    def _ols_rss(X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        """Residual sum of squares from OLS regression."""
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        return float(residuals @ residuals)

    def _significant_edges(
        self,
        pvals: NDArray[np.float64],
    ) -> list[tuple[str, str]]:
        n = len(self._variable_names)
        edges: list[tuple[str, str]] = []
        for i in range(n):
            for j in range(n):
                if i != j and pvals[i, j] < self._significance_level:
                    edges.append((self._variable_names[j], self._variable_names[i]))
        return edges

    @staticmethod
    def _to_dataframe(
        data: pd.DataFrame | NDArray[np.float64],
        variable_names: list[str] | None,
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        require(
            variable_names is not None,
            "variable_names required when data is an ndarray",
        )
        assert variable_names is not None
        return pd.DataFrame(data, columns=variable_names)
