"""Tests for the MarketModel abstract base class."""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.models.base import MarketModel


class _DummyModel(MarketModel):
    """Minimal concrete subclass for testing the ABC interface."""

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def n_assets(self) -> int:
        return self._n

    def drift_rates(self, t, x):  # type: ignore[override]
        return np.zeros(self._n)

    def covariance_rate(self, t, x):  # type: ignore[override]
        return np.eye(self._n) * 0.04

    def to_stochastic_process(self, x0):  # type: ignore[override]
        from quantspt.core.processes import CorrelatedGBM

        return CorrelatedGBM(
            mu=np.zeros(self._n),
            cov=np.eye(self._n) * 0.04,
            x0=x0,
        )


class TestMarketModelABC:
    """Verify ABC contract and default market_weights method."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            MarketModel()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        model = _DummyModel(3)
        assert model.n_assets == 3

    def test_market_weights_sum_to_one(self) -> None:
        model = _DummyModel(4)
        x = np.array([100.0, 200.0, 300.0, 400.0])
        mu = model.market_weights(x)
        np.testing.assert_allclose(mu.sum(), 1.0)

    def test_market_weights_proportional(self) -> None:
        model = _DummyModel(3)
        x = np.array([10.0, 30.0, 60.0])
        mu = model.market_weights(x)
        np.testing.assert_allclose(mu, [0.1, 0.3, 0.6])

    def test_market_weights_single_stock(self) -> None:
        model = _DummyModel(1)
        mu = model.market_weights(np.array([42.0]))
        np.testing.assert_allclose(mu, [1.0])
