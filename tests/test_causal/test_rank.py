"""Tests for causal rank dynamics analysis.

Validates Granger-style causal testing on rank time series, causal
edge detection, and integration with market weight paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantspt.causal.rank import CausalRankAnalysis
from quantspt.errors import SPTInvariantError


@pytest.fixture()
def causal_rank_series() -> pd.DataFrame:
    """Synthetic rank changes where A Granger-causes B.

    A is autoregressive; B follows A with a lag.
    C is independent noise.
    """
    rng = np.random.default_rng(42)
    T = 500
    A = np.zeros(T)
    B = np.zeros(T)
    C = rng.normal(size=T)

    for t in range(1, T):
        A[t] = 0.5 * A[t - 1] + rng.normal() * 0.5
        B[t] = 0.7 * A[t - 1] + 0.2 * B[t - 1] + rng.normal() * 0.3

    return pd.DataFrame({"A": A, "B": B, "C": C})


@pytest.fixture()
def independent_series() -> pd.DataFrame:
    """Three independent random walks (no Granger causality)."""
    rng = np.random.default_rng(42)
    T = 300
    return pd.DataFrame(
        {
            "X": rng.normal(size=T),
            "Y": rng.normal(size=T),
            "Z": rng.normal(size=T),
        }
    )


@pytest.fixture()
def synthetic_weights() -> np.ndarray:
    """Market weight paths for 4 stocks over 200 timesteps."""
    rng = np.random.default_rng(42)
    T = 200
    n = 4
    raw = rng.dirichlet(np.ones(n), size=T)
    for t in range(1, T):
        raw[t] = 0.95 * raw[t - 1] + 0.05 * raw[t]
        raw[t] /= raw[t].sum()
    return raw


class TestGrangerCausality:
    """Pairwise Granger causality testing."""

    def test_detects_causal_direction(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        """A should Granger-cause B, but not vice versa."""
        cra = CausalRankAnalysis(max_lag=3, significance_level=0.05)
        cra.fit(causal_rank_series)

        pvals = cra.granger_pvalues
        names = cra.variable_names
        idx = {n: i for i, n in enumerate(names)}

        assert pvals[idx["B"], idx["A"]] < 0.05
        assert pvals[idx["A"], idx["B"]] > 0.05

    def test_independent_series_no_causality(
        self,
        independent_series: pd.DataFrame,
    ) -> None:
        cra = CausalRankAnalysis(max_lag=3, significance_level=0.01)
        cra.fit(independent_series)

        assert len(cra.causal_edges) == 0

    def test_pvalue_matrix_shape(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        cra = CausalRankAnalysis(max_lag=3)
        cra.fit(causal_rank_series)
        pvals = cra.granger_pvalues
        assert pvals.shape == (3, 3)

    def test_diagonal_is_one(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        """Self-causality p-values should remain at 1.0 (untested)."""
        cra = CausalRankAnalysis(max_lag=3)
        cra.fit(causal_rank_series)
        pvals = cra.granger_pvalues
        np.testing.assert_allclose(np.diag(pvals), 1.0)

    def test_fstats_shape(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        cra = CausalRankAnalysis(max_lag=3)
        cra.fit(causal_rank_series)
        fstats = cra.granger_fstats
        assert fstats.shape == (3, 3)
        assert np.all(fstats >= 0)

    def test_pvalues_bounded(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        cra = CausalRankAnalysis(max_lag=3)
        cra.fit(causal_rank_series)
        pvals = cra.granger_pvalues
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)


class TestCausalEdges:
    """Causal edge detection at specified significance level."""

    def test_edges_contain_true_cause(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        cra = CausalRankAnalysis(max_lag=3, significance_level=0.05)
        cra.fit(causal_rank_series)
        assert ("A", "B") in cra.causal_edges

    def test_edge_format(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        cra = CausalRankAnalysis(max_lag=3, significance_level=0.05)
        cra.fit(causal_rank_series)
        for cause, effect in cra.causal_edges:
            assert isinstance(cause, str)
            assert isinstance(effect, str)


class TestCausalStrength:
    """Causal strength matrix (-log10(p))."""

    def test_strength_shape(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        cra = CausalRankAnalysis(max_lag=3)
        cra.fit(causal_rank_series)
        strength = cra.causal_strength()
        assert strength.shape == (3, 3)

    def test_strength_diagonal_zero(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        cra = CausalRankAnalysis(max_lag=3)
        cra.fit(causal_rank_series)
        strength = cra.causal_strength()
        np.testing.assert_allclose(np.diag(strength), 0.0)

    def test_strength_nonnegative(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        cra = CausalRankAnalysis(max_lag=3)
        cra.fit(causal_rank_series)
        strength = cra.causal_strength()
        assert np.all(strength >= 0)

    def test_stronger_for_true_cause(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        cra = CausalRankAnalysis(max_lag=3)
        cra.fit(causal_rank_series)
        strength = cra.causal_strength()
        names = cra.variable_names
        idx = {n: i for i, n in enumerate(names)}

        assert strength[idx["B"], idx["A"]] > strength[idx["A"], idx["B"]]


class TestFromWeightPaths:
    """Integration with market weight paths."""

    def test_from_weight_paths(self, synthetic_weights: np.ndarray) -> None:
        cra = CausalRankAnalysis.from_weight_paths(
            synthetic_weights,
            max_lag=3,
        )
        assert len(cra.variable_names) == 4
        assert cra.granger_pvalues.shape == (4, 4)

    def test_from_weight_paths_custom_names(
        self, synthetic_weights: np.ndarray
    ) -> None:
        names = ["AAPL", "GOOG", "MSFT", "AMZN"]
        cra = CausalRankAnalysis.from_weight_paths(
            synthetic_weights,
            stock_names=names,
            max_lag=3,
        )
        assert cra.variable_names == names

    def test_from_weight_paths_default_names(
        self, synthetic_weights: np.ndarray
    ) -> None:
        cra = CausalRankAnalysis.from_weight_paths(
            synthetic_weights,
            max_lag=3,
        )
        assert cra.variable_names[0] == "stock_0"


class TestConfiguration:
    """Configurable parameters."""

    def test_max_lag(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        for lag in (1, 3, 5):
            cra = CausalRankAnalysis(max_lag=lag)
            cra.fit(causal_rank_series)
            assert len(cra.causal_edges) >= 0

    def test_significance_level_strictness(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        """Stricter significance → fewer edges."""
        cra_loose = CausalRankAnalysis(max_lag=3, significance_level=0.10)
        cra_loose.fit(causal_rank_series)

        cra_strict = CausalRankAnalysis(max_lag=3, significance_level=0.001)
        cra_strict.fit(causal_rank_series)

        assert len(cra_strict.causal_edges) <= len(cra_loose.causal_edges)


class TestValidation:
    """Input validation."""

    def test_invalid_max_lag(self) -> None:
        with pytest.raises(SPTInvariantError, match="max_lag"):
            CausalRankAnalysis(max_lag=0)

    def test_access_before_fit(self) -> None:
        cra = CausalRankAnalysis()
        with pytest.raises(SPTInvariantError, match="fit"):
            _ = cra.granger_pvalues
        with pytest.raises(SPTInvariantError, match="fit"):
            _ = cra.causal_edges
        with pytest.raises(SPTInvariantError, match="fit"):
            cra.causal_strength()

    def test_ndarray_input(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(size=(100, 3))
        cra = CausalRankAnalysis(max_lag=2)
        cra.fit(data, variable_names=["X", "Y", "Z"])
        assert cra.granger_pvalues.shape == (3, 3)

    def test_ndarray_without_names_raises(self) -> None:
        data = np.random.default_rng(42).normal(size=(100, 3))
        cra = CausalRankAnalysis()
        with pytest.raises(SPTInvariantError, match="variable_names"):
            cra.fit(data)

    def test_fit_returns_self(
        self,
        causal_rank_series: pd.DataFrame,
    ) -> None:
        cra = CausalRankAnalysis(max_lag=3)
        result = cra.fit(causal_rank_series)
        assert result is cra
