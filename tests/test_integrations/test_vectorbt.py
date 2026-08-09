"""Tests for the vectorbt / backtrader integration adapter.

Tests focus on the signal generation logic (which works without
vectorbt installed). The actual vectorbt Portfolio creation is only
tested if vectorbt is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantspt.core.generating_functions import (
    DiversityGenerator,
    EntropyGenerator,
)
from quantspt.integrations.vectorbt import (
    SPTBacktraderStrategy,
    SPTSignalFactory,
)


@pytest.fixture()
def price_df() -> pd.DataFrame:
    """Synthetic price DataFrame for 5 assets, 100 days."""
    rng = np.random.default_rng(42)
    n, T = 5, 100
    prices = np.zeros((T, n))
    prices[0] = [100, 150, 80, 200, 120]
    for t in range(1, T):
        returns = 1.0 + rng.normal(0, 0.02, n)
        prices[t] = prices[t - 1] * returns

    return pd.DataFrame(
        prices,
        columns=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        index=pd.date_range("2024-01-01", periods=T, freq="B"),
    )


@pytest.fixture()
def price_array(price_df: pd.DataFrame) -> np.ndarray:
    return price_df.values


class TestSPTSignalFactory:
    def test_init_with_generating_function(self) -> None:
        gen = DiversityGenerator(0.3)
        factory = SPTSignalFactory(gen)
        assert factory.name == gen.name

    def test_init_with_weight_func(self) -> None:
        factory = SPTSignalFactory(weight_func=lambda mu: mu**0.5 / (mu**0.5).sum())
        assert factory.name == "CustomWeightFunc"

    def test_init_neither_raises(self) -> None:
        with pytest.raises(ValueError, match="Provide either"):
            SPTSignalFactory()

    def test_generate_signals_from_dataframe(self, price_df: pd.DataFrame) -> None:
        gen = DiversityGenerator(0.3)
        factory = SPTSignalFactory(gen, rebalance_every=21)
        signals = factory.generate_signals(price_df)

        assert isinstance(signals, pd.DataFrame)
        assert signals.shape == price_df.shape
        assert list(signals.columns) == list(price_df.columns)

        for t in range(len(signals)):
            row = signals.iloc[t].values
            assert abs(row.sum() - 1.0) < 1e-8
            assert np.all(row >= 0)

    def test_generate_signals_from_array(self, price_array: np.ndarray) -> None:
        gen = DiversityGenerator(0.5)
        factory = SPTSignalFactory(gen, rebalance_every=10)
        signals = factory.generate_signals(price_array)

        assert isinstance(signals, pd.DataFrame)
        assert signals.shape == price_array.shape

    def test_rebalance_frequency(self, price_df: pd.DataFrame) -> None:
        """Weights only change on rebalance days."""
        gen = DiversityGenerator(0.3)
        factory = SPTSignalFactory(gen, rebalance_every=21)
        signals = factory.generate_signals(price_df)

        w0 = signals.iloc[0].values
        w1 = signals.iloc[1].values
        assert np.allclose(w0, w1)

        w21 = signals.iloc[21].values
        assert not np.allclose(w0, w21)

    def test_different_generators(self, price_df: pd.DataFrame) -> None:
        """Works with different generating functions."""
        for gen in [DiversityGenerator(0.3), DiversityGenerator(0.7)]:
            factory = SPTSignalFactory(gen)
            signals = factory.generate_signals(price_df)
            assert signals.shape == price_df.shape

    def test_initial_weights_override(self, price_df: pd.DataFrame) -> None:
        n = price_df.shape[1]
        equal_w = np.ones(n) / n
        factory = SPTSignalFactory(DiversityGenerator(0.3))
        signals = factory.generate_signals(price_df, initial_weights=equal_w)
        assert signals.shape == price_df.shape

    def test_entropy_generator(self, price_df: pd.DataFrame) -> None:
        gen = EntropyGenerator()
        factory = SPTSignalFactory(gen)
        signals = factory.generate_signals(price_df)
        assert signals.shape == price_df.shape

    def test_vectorbt_requires_import(self, price_df: pd.DataFrame) -> None:
        """to_vectorbt_signals raises if vectorbt not installed."""
        gen = DiversityGenerator(0.3)
        factory = SPTSignalFactory(gen)
        try:
            import vectorbt  # noqa: F401

            result = factory.to_vectorbt_signals(price_df)
            assert result is not None
        except ImportError:
            with pytest.raises(ImportError, match="vectorbt"):
                factory.to_vectorbt_signals(price_df)


class TestSPTBacktraderStrategy:
    def test_init(self) -> None:
        gen = DiversityGenerator(0.5)
        adapter = SPTBacktraderStrategy(gen, rebalance_every=21)
        assert adapter is not None

    def test_strategy_class_requires_backtrader(self) -> None:
        gen = DiversityGenerator(0.3)
        adapter = SPTBacktraderStrategy(gen)
        try:
            import backtrader  # noqa: F401

            cls = adapter.strategy_class()
            assert cls is not None
        except ImportError:
            with pytest.raises(ImportError, match="backtrader"):
                adapter.strategy_class()
