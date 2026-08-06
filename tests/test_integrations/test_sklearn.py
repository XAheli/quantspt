"""Tests for integrations/sklearn.py — sklearn bridge transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantspt.errors import SPTInvariantError
from quantspt.integrations.sklearn import (
    DiversityFeature,
    ExcessGrowthFeature,
    SPTTransformer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def price_matrix():
    rng = np.random.default_rng(42)
    return rng.uniform(50, 150, (100, 5))


@pytest.fixture()
def weight_matrix():
    rng = np.random.default_rng(42)
    w = rng.dirichlet(np.ones(5), size=50)
    return w


# ---------------------------------------------------------------------------
# SPTTransformer tests
# ---------------------------------------------------------------------------


class TestSPTTransformer:
    def test_basic_transform(self, price_matrix) -> None:
        t = SPTTransformer()
        result = t.fit_transform(price_matrix)
        assert result.shape == price_matrix.shape
        assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_weights_positive(self, price_matrix) -> None:
        t = SPTTransformer()
        result = t.transform(price_matrix)
        assert np.all(result >= 0)

    def test_fit_returns_self(self) -> None:
        t = SPTTransformer()
        result = t.fit(np.ones((5, 3)))
        assert result is t

    def test_min_weight_filters(self) -> None:
        prices = np.array([[1000.0, 1.0, 1.0]])
        t = SPTTransformer(min_weight=0.01)
        result = t.transform(prices)
        assert result[0, 1] == 0.0
        assert result[0, 2] == 0.0
        assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_dataframe_input(self) -> None:
        df = pd.DataFrame({"A": [100.0, 110.0], "B": [200.0, 190.0]})
        t = SPTTransformer()
        result = t.transform(df)
        assert result.shape == (2, 2)
        assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_get_set_params(self) -> None:
        t = SPTTransformer(normalize=False, min_weight=0.05)
        params = t.get_params()
        assert params == {"normalize": False, "min_weight": 0.05}
        t.set_params(min_weight=0.1)
        assert t.min_weight == 0.1

    def test_repr(self) -> None:
        t = SPTTransformer()
        assert "SPTTransformer" in repr(t)
        assert "normalize=True" in repr(t)

    def test_1d_input_raises(self) -> None:
        t = SPTTransformer()
        with pytest.raises(SPTInvariantError, match="2-D"):
            t.transform(np.ones(10))

    def test_equal_prices_equal_weights(self) -> None:
        prices = np.ones((5, 4)) * 100.0
        t = SPTTransformer()
        result = t.transform(prices)
        assert_allclose(result, 0.25, atol=1e-10)


# ---------------------------------------------------------------------------
# DiversityFeature tests
# ---------------------------------------------------------------------------


class TestDiversityFeature:
    def test_output_shape(self, weight_matrix) -> None:
        feat = DiversityFeature(p=0.5, from_weights=True)
        result = feat.fit_transform(weight_matrix)
        assert result.shape == (50, 1)

    def test_diversity_bounded(self, weight_matrix) -> None:
        feat = DiversityFeature(p=0.5, from_weights=True)
        result = feat.fit_transform(weight_matrix)
        n = weight_matrix.shape[1]
        assert np.all(result >= 1.0 - 1e-10)
        assert np.all(result <= n ** (1.0 / 0.5 - 1.0) + 1e-5)

    def test_equal_weights_maximum_diversity(self) -> None:
        n = 5
        weights = np.ones((10, n)) / n
        feat = DiversityFeature(p=0.5, from_weights=True)
        result = feat.transform(weights)
        expected = n ** (1.0 / 0.5 - 1.0)
        assert_allclose(result[0, 0], expected, atol=1e-8)

    def test_concentrated_minimum_diversity(self) -> None:
        weights = np.zeros((5, 4))
        weights[:, 0] = 1.0
        feat = DiversityFeature(p=0.5, from_weights=True)
        result = feat.transform(weights)
        assert_allclose(result[:, 0], 1.0, atol=1e-6)

    def test_from_prices(self, price_matrix) -> None:
        feat = DiversityFeature(p=0.5, from_weights=False)
        result = feat.transform(price_matrix)
        assert result.shape == (100, 1)
        assert np.all(np.isfinite(result))

    def test_p_one_always_one(self, weight_matrix) -> None:
        feat = DiversityFeature(p=1.0, from_weights=True)
        result = feat.transform(weight_matrix)
        assert_allclose(result, 1.0, atol=1e-10)

    def test_get_set_params(self) -> None:
        feat = DiversityFeature(p=0.7, from_weights=True)
        params = feat.get_params()
        assert params == {"p": 0.7, "from_weights": True}
        feat.set_params(p=0.3)
        assert feat.p == 0.3

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="Diversity parameter"):
            DiversityFeature(p=0.0)

    def test_repr(self) -> None:
        feat = DiversityFeature(p=0.5)
        assert "DiversityFeature" in repr(feat)
        assert "0.5" in repr(feat)

    def test_dataframe_input(self) -> None:
        df = pd.DataFrame({"A": [0.3, 0.4], "B": [0.7, 0.6]})
        feat = DiversityFeature(p=0.5, from_weights=True)
        result = feat.transform(df)
        assert result.shape == (2, 1)


# ---------------------------------------------------------------------------
# ExcessGrowthFeature tests
# ---------------------------------------------------------------------------


class TestExcessGrowthFeature:
    def test_output_shape(self, price_matrix) -> None:
        feat = ExcessGrowthFeature(window=20, min_periods=10)
        result = feat.fit_transform(price_matrix)
        assert result.shape == (100, 1)

    def test_initial_nan(self, price_matrix) -> None:
        feat = ExcessGrowthFeature(window=20, min_periods=15)
        result = feat.transform(price_matrix)
        assert np.all(np.isnan(result[:15, 0]))

    def test_non_negative_for_long_only(self, price_matrix) -> None:
        feat = ExcessGrowthFeature(window=30, min_periods=10)
        result = feat.transform(price_matrix)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1e-10)

    def test_from_weights(self, weight_matrix) -> None:
        feat = ExcessGrowthFeature(window=10, from_weights=True, min_periods=5)
        result = feat.transform(weight_matrix)
        assert result.shape == (50, 1)

    def test_single_asset_zero_gamma(self) -> None:
        prices = np.arange(100, 200, dtype=np.float64).reshape(-1, 1)
        feat = ExcessGrowthFeature(window=10, min_periods=5)
        result = feat.transform(prices)
        valid = result[~np.isnan(result)]
        assert_allclose(valid, 0.0, atol=1e-10)

    def test_get_set_params(self) -> None:
        feat = ExcessGrowthFeature(window=30, min_periods=10)
        params = feat.get_params()
        assert params == {"window": 30, "from_weights": False, "min_periods": 10}
        feat.set_params(window=60)
        assert feat.window == 60

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="window"):
            ExcessGrowthFeature(window=1)

    def test_invalid_min_periods_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="min_periods"):
            ExcessGrowthFeature(min_periods=1)

    def test_repr(self) -> None:
        feat = ExcessGrowthFeature(window=60)
        assert "ExcessGrowthFeature" in repr(feat)
        assert "60" in repr(feat)

    def test_dataframe_input(self, price_matrix) -> None:
        df = pd.DataFrame(price_matrix, columns=[f"S{i}" for i in range(5)])
        feat = ExcessGrowthFeature(window=20, min_periods=10)
        result = feat.transform(df)
        assert result.shape == (100, 1)
