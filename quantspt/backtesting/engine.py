"""Event-driven backtesting engine for SPT strategies.

The engine simulates portfolio evolution under a generating-function-based
strategy (or any custom weight function), with configurable rebalancing
triggers and execution models. It produces a complete time series of
portfolio values, weights, turnover, and transaction costs.

Mathematical References
-----------------------
- Portfolio value evolution: V(t+dt) = V(t) · Σ_i π_i(t) · R_i(t)
- Turnover: T(t) = Σ_i |π_i(t) - π_i(t-)|  where π(t-) is pre-rebalance
- Master formula: F&K Survey Eq. 11.2
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._preconditions import ensure, require
from .._result import SPTResult
from .execution import ExecutionModel, ExecutionResult, InstantExecution
from .rebalancing import CalendarRebalancer, Frequency, Rebalancer

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
]


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a backtest run.

    Parameters
    ----------
    initial_value : float
        Starting portfolio value.
    dt : float
        Time step in years (e.g. 1/252 for daily).
    """

    initial_value: float = 1.0
    dt: float = 1.0 / 252.0


@dataclass
class BacktestResult:
    """Complete backtest output with time series data.

    All arrays are indexed by time step [0, T].

    Attributes
    ----------
    portfolio_values : ndarray of shape (T+1,)
        Portfolio value at each time step.
    market_values : ndarray of shape (T+1,)
        Market portfolio value at each time step.
    weights_history : ndarray of shape (T+1, n)
        Portfolio weights at each time step (post-rebalance if applicable).
    market_weights_history : ndarray of shape (T+1, n)
        Market weights at each time step.
    turnover : ndarray of shape (T+1,)
        Turnover at each time step (0 when no rebalance).
    costs : ndarray of shape (T+1,)
        Transaction cost fraction at each time step.
    rebalance_steps : list of int
        Steps where rebalancing occurred.
    n_rebalances : int
        Total number of rebalances.
    config : BacktestConfig
        Configuration used.
    """

    portfolio_values: NDArray[np.float64]
    market_values: NDArray[np.float64]
    weights_history: NDArray[np.float64]
    market_weights_history: NDArray[np.float64]
    turnover: NDArray[np.float64]
    costs: NDArray[np.float64]
    rebalance_steps: list[int]
    n_rebalances: int
    config: BacktestConfig

    @property
    def n_steps(self) -> int:
        """Number of time steps in the backtest."""
        return len(self.portfolio_values) - 1

    def log_relative_return(self) -> float:
        """Compute log(V^π(T) / V^μ(T))."""
        return float(np.log(self.portfolio_values[-1] / self.market_values[-1]))

    def total_turnover(self) -> float:
        """Sum of all turnover across the backtest."""
        return float(np.sum(self.turnover))

    def total_cost(self) -> float:
        """Sum of all transaction costs (as fraction of value)."""
        return float(np.sum(self.costs))


WeightFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class BacktestEngine:
    """Event-driven backtesting engine for SPT strategies.

    Takes a weight function (from a GeneratingFunction or custom callable),
    a returns series, and optional rebalancing/execution models. Runs an
    event-driven loop that tracks portfolio evolution.

    Parameters
    ----------
    weight_func : callable
        Function mapping market weights (n,) → portfolio weights (n,).
        Typically ``G.weights`` for a ``GeneratingFunction`` G.
    returns : ndarray of shape (T, n)
        Asset return series. Each row is (1 + r_i) for that period.
    initial_weights : ndarray of shape (n,)
        Initial market weights at time 0.
    rebalancer : Rebalancer, optional
        Rebalancing trigger. Defaults to monthly calendar rebalancing.
    execution : ExecutionModel, optional
        Execution model. Defaults to instant (zero-cost) execution.
    config : BacktestConfig, optional
        Backtest configuration.
    """

    def __init__(
        self,
        weight_func: WeightFunction,
        returns: NDArray[np.float64],
        initial_weights: NDArray[np.float64],
        rebalancer: Rebalancer | None = None,
        execution: ExecutionModel | None = None,
        config: BacktestConfig | None = None,
    ) -> None:
        require(returns.ndim == 2, "returns must be 2-D (T, n)")
        require(
            bool(np.all(np.isfinite(returns))),
            "returns must not contain NaN or Inf",
        )
        require(
            len(initial_weights) == returns.shape[1], "initial_weights/returns mismatch"
        )
        require(
            bool(np.all(initial_weights > 0)),
            "initial_weights must be positive",
        )
        require(
            bool(abs(float(np.sum(initial_weights)) - 1.0) < 1e-8),
            "initial_weights must sum to 1",
        )

        self._weight_func = weight_func
        self._returns = returns
        self._initial_weights = initial_weights
        self._rebalancer: Rebalancer = rebalancer or CalendarRebalancer(
            Frequency.MONTHLY
        )
        self._execution: ExecutionModel = execution or InstantExecution()
        self._config = config or BacktestConfig()

    def run(self) -> SPTResult[BacktestResult]:
        """Execute the backtest.

        Returns
        -------
        SPTResult[BacktestResult]
            Complete backtest results wrapped in the standard envelope.
        """
        start_time = time.perf_counter()

        T = len(self._returns)
        n = self._returns.shape[1]
        cfg = self._config

        portfolio_values = np.zeros(T + 1)
        market_values = np.zeros(T + 1)
        weights_history = np.zeros((T + 1, n))
        market_weights_history = np.zeros((T + 1, n))
        turnover_arr = np.zeros(T + 1)
        costs_arr = np.zeros(T + 1)
        rebalance_steps: list[int] = []

        portfolio_values[0] = cfg.initial_value
        market_values[0] = cfg.initial_value
        mu_t = self._initial_weights.copy()
        market_weights_history[0] = mu_t

        target_weights = self._weight_func(mu_t)
        target_weights = np.clip(target_weights, 0.0, None)
        w_sum = float(np.sum(target_weights))
        if w_sum > 0:
            target_weights = target_weights / w_sum

        exec_result: ExecutionResult = self._execution.execute(
            mu_t, target_weights, portfolio_values[0]
        )
        pi_t = exec_result.weights
        costs_arr[0] = exec_result.cost
        turnover_arr[0] = float(np.sum(np.abs(pi_t - mu_t)))
        rebalance_steps.append(0)

        portfolio_values[0] *= 1.0 - exec_result.cost
        weights_history[0] = pi_t

        for t in range(T):
            ret_t = self._returns[t]

            new_port_val = portfolio_values[t] * float(np.dot(pi_t, ret_t))
            portfolio_values[t + 1] = new_port_val

            new_mkt_val = market_values[t] * float(np.dot(mu_t, ret_t))
            market_values[t + 1] = new_mkt_val

            raw_pi = pi_t * ret_t
            pi_sum = float(np.sum(raw_pi))
            pi_drifted = raw_pi / pi_sum if pi_sum > 0 else pi_t.copy()

            raw_mu = mu_t * ret_t
            mu_sum = float(np.sum(raw_mu))
            mu_t = raw_mu / mu_sum if mu_sum > 0 else mu_t.copy()
            market_weights_history[t + 1] = mu_t

            target_weights = self._weight_func(mu_t)
            target_weights = np.clip(target_weights, 0.0, None)
            w_sum = float(np.sum(target_weights))
            if w_sum > 0:
                target_weights = target_weights / w_sum

            if self._rebalancer.should_rebalance(t + 1, pi_drifted, target_weights):
                exec_result = self._execution.execute(
                    pi_drifted, target_weights, portfolio_values[t + 1]
                )
                pi_t = exec_result.weights
                costs_arr[t + 1] = exec_result.cost
                turnover_arr[t + 1] = float(np.sum(np.abs(pi_t - pi_drifted)))
                portfolio_values[t + 1] *= 1.0 - exec_result.cost
                rebalance_steps.append(t + 1)
            else:
                pi_t = pi_drifted

            weights_history[t + 1] = pi_t

        ensure(
            bool(np.all(np.isfinite(portfolio_values))),
            "Portfolio values contain non-finite entries",
        )

        result = BacktestResult(
            portfolio_values=portfolio_values,
            market_values=market_values,
            weights_history=weights_history,
            market_weights_history=market_weights_history,
            turnover=turnover_arr,
            costs=costs_arr,
            rebalance_steps=rebalance_steps,
            n_rebalances=len(rebalance_steps),
            config=cfg,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SPTResult(
            data=result,
            metadata={
                "engine": "BacktestEngine",
                "n_steps": T,
                "n_assets": n,
                "n_rebalances": len(rebalance_steps),
                "total_turnover": result.total_turnover(),
                "total_cost": result.total_cost(),
            },
            computation_time_ms=elapsed_ms,
        )
