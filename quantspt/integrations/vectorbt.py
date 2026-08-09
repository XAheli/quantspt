"""Adapter for vectorbt and backtrader integration.

Provides bridge classes that let users run SPT generating-function
strategies inside vectorbt's signal/portfolio framework or
backtrader's Strategy class.

Requires ``vectorbt`` or ``backtrader`` to be installed separately.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..core.generating_functions import GeneratingFunction

__all__ = [
    "SPTBacktraderStrategy",
    "SPTSignalFactory",
]


def _require_vectorbt() -> Any:
    try:
        import vectorbt as vbt

        return vbt
    except ImportError as exc:
        raise ImportError(
            "vectorbt is required for this adapter. Install with: pip install vectorbt"
        ) from exc


def _require_backtrader() -> Any:
    try:
        import backtrader as bt

        return bt
    except ImportError as exc:
        raise ImportError(
            "backtrader is required for this adapter. "
            "Install with: pip install backtrader"
        ) from exc


class SPTSignalFactory:
    """Adapter that generates vectorbt-compatible allocation signals.

    Converts an SPT generating function's weight output into a
    time series of target allocations suitable for vectorbt's
    ``Portfolio.from_orders`` or signal-based workflows.

    Parameters
    ----------
    generating_function : GeneratingFunction
        Any SPT generating function (DiversityGenerator, EntropyGenerator,
        CovarianceConditionalFGP, etc.).
    rebalance_every : int
        Rebalance every N bars (trading days). Default 21 (monthly).

    Examples
    --------
    ::

        gen = DiversityGenerator(p=0.3)
        factory = SPTSignalFactory(gen, rebalance_every=21)

        # From a price DataFrame
        target_weights = factory.generate_signals(price_df)

        # Use with vectorbt (if installed)
        # pf = vbt.Portfolio.from_orders(close=price_df, size=target_weights)
    """

    def __init__(
        self,
        generating_function: GeneratingFunction | None = None,
        *,
        weight_func: (
            Callable[[NDArray[np.float64]], NDArray[np.float64]] | None
        ) = None,
        rebalance_every: int = 21,
    ) -> None:
        if generating_function is not None:
            self._weight_func: Callable[[NDArray[np.float64]], NDArray[np.float64]] = (
                generating_function.weights
            )
            self._name = generating_function.name
        elif weight_func is not None:
            self._weight_func = weight_func
            self._name = "CustomWeightFunc"
        else:
            raise ValueError("Provide either generating_function or weight_func")
        self._rebalance_every = rebalance_every

    @property
    def name(self) -> str:
        """Strategy name for display."""
        return self._name

    def generate_signals(
        self,
        prices: pd.DataFrame | NDArray[np.float64],
        *,
        initial_weights: NDArray[np.float64] | None = None,
    ) -> pd.DataFrame:
        """Generate target weight signals from a price series.

        Parameters
        ----------
        prices : DataFrame or ndarray of shape (T, n)
            Historical prices. If a DataFrame, column names are preserved.
        initial_weights : ndarray of shape (n,), optional
            Starting market weights. If None, computed from first row of
            prices (market-cap proportional).

        Returns
        -------
        DataFrame of shape (T, n)
            Target portfolio weights at each time step. Weights are only
            updated on rebalance days; in between they carry forward.
        """
        if isinstance(prices, pd.DataFrame):
            columns = prices.columns
            index: Any = prices.index
            price_arr: NDArray[np.float64] = prices.to_numpy(dtype=np.float64)
        else:
            price_arr = np.asarray(prices, dtype=np.float64)
            columns = pd.Index([f"asset_{i}" for i in range(price_arr.shape[1])])
            index = pd.RangeIndex(price_arr.shape[0])

        T, n = price_arr.shape
        weights_out = np.zeros((T, n))

        if initial_weights is not None:
            mu = initial_weights.copy()
        else:
            row0 = price_arr[0]
            mu = row0 / row0.sum()

        target = self._weight_func(mu)
        target = np.clip(target, 0.0, None)
        s = target.sum()
        if s > 0:
            target /= s
        weights_out[0] = target

        for t in range(1, T):
            row = price_arr[t]
            row_prev = price_arr[t - 1]
            valid = row_prev > 0
            ret = np.ones(n)
            ret[valid] = row[valid] / row_prev[valid]

            raw_mu = mu * ret
            mu_sum = raw_mu.sum()
            mu = raw_mu / mu_sum if mu_sum > 0 else mu

            if t % self._rebalance_every == 0:
                target = self._weight_func(mu)
                target = np.clip(target, 0.0, None)
                s = target.sum()
                if s > 0:
                    target /= s

            weights_out[t] = target

        return pd.DataFrame(weights_out, index=index, columns=columns)

    def to_vectorbt_signals(
        self,
        prices: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        """Generate signals and create a vectorbt Portfolio.

        Parameters
        ----------
        prices : DataFrame
            Price data.
        **kwargs
            Forwarded to ``vbt.Portfolio.from_orders``.

        Returns
        -------
        vbt.Portfolio
        """
        vbt = _require_vectorbt()
        weights = self.generate_signals(prices, **kwargs)

        return vbt.Portfolio.from_orders(
            close=prices,
            size=weights,
            size_type="targetpercent",
            group_by=True,
            cash_sharing=True,
        )


class SPTBacktraderStrategy:
    """Configuration wrapper for a backtrader Strategy using SPT weights.

    This is a factory that produces a backtrader ``bt.Strategy`` subclass
    configured with an SPT generating function. The user instantiates
    this, then passes the result of ``strategy_class()`` to
    ``cerebro.addstrategy()``.

    Parameters
    ----------
    generating_function : GeneratingFunction
        SPT generating function to use.
    rebalance_every : int
        Bars between rebalances (default 21 = monthly).

    Examples
    --------
    ::

        adapter = SPTBacktraderStrategy(DiversityGenerator(p=0.3))
        StrategyClass = adapter.strategy_class()

        cerebro = bt.Cerebro()
        cerebro.addstrategy(StrategyClass)
    """

    def __init__(
        self,
        generating_function: GeneratingFunction,
        rebalance_every: int = 21,
    ) -> None:
        self._gen = generating_function
        self._rebalance_every = rebalance_every

    def strategy_class(self) -> type:
        """Return a backtrader Strategy subclass.

        The class has the generating function and rebalancing interval
        baked in.
        """
        bt = _require_backtrader()
        gen = self._gen
        rebal_every = self._rebalance_every

        class _SPTStrategy(bt.Strategy):  # type: ignore[misc,name-defined]
            params = (("rebalance_every", rebal_every),)

            def __init__(self) -> None:
                self._bar = 0

            def next(self) -> None:
                self._bar += 1
                if self._bar % self.params.rebalance_every != 0:  # type: ignore[attr-defined]
                    return

                feeds = self.datas  # codespell:ignore datas
                prices = np.array(
                    [d.close[0] for d in feeds],
                    dtype=np.float64,
                )
                total = prices.sum()
                if total <= 0:
                    return
                mu = prices / total

                target = gen.weights(mu)
                target = np.clip(target, 0.0, None)
                s = target.sum()
                if s > 0:
                    target /= s

                for i, feed in enumerate(feeds):
                    self.order_target_percent(feed, target=float(target[i]))  # type: ignore[attr-defined]

        _SPTStrategy.__name__ = f"SPT_{gen.name}"
        _SPTStrategy.__qualname__ = _SPTStrategy.__name__
        return _SPTStrategy
