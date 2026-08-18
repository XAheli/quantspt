"""Market-level simulation: prices, weights, and ranks over time.

Given a ``MarketModel``, simulates price paths and derives weight paths
and rank dynamics, including local time tracking for rank-based models.

Mathematical References
-----------------------
- Market weights μ_i = X_i / Σ X_j: F&K Survey Eq. 1.2
- Rank assignment: F&K Survey Eq. 1.18
- Local time accumulation at rank collisions: BFK §3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from .._result import SPTResult, timed_result
from ..core.processes import simulate_path

__all__ = [
    "MarketSimulation",
    "simulate_market",
]


@dataclass
class MarketSimulation:
    """Result of a market simulation.

    Attributes
    ----------
    times : ndarray of shape (n_steps+1,)
        Time grid.
    prices : ndarray of shape (n_steps+1, n_assets)
        Simulated price paths.
    weights : ndarray of shape (n_steps+1, n_assets)
        Market-capitalisation weights at each time step.
    ranks : ndarray of shape (n_steps+1, n_assets), dtype intp
        Rank of each asset at each time step (0 = largest).
    local_times : ndarray of shape (n_steps+1, n_assets-1) or None
        Cumulative local time at each adjacent rank boundary.
        ``None`` if local time tracking was not requested.
    """

    times: NDArray[np.float64]
    prices: NDArray[np.float64]
    weights: NDArray[np.float64]
    ranks: NDArray[np.intp]
    local_times: NDArray[np.float64] | None = None


def simulate_market(
    model: object,  # MarketModel
    x0: NDArray[np.float64],
    T: float,
    n_steps: int,
    seed: int | None = None,
    discretization: object | None = None,
    track_local_times: bool = False,
    local_time_epsilon: float | None = None,
) -> SPTResult[MarketSimulation]:
    r"""Simulate a market model and compute weights and ranks.

    Converts a ``MarketModel`` to a ``StochasticProcess`` via
    ``to_stochastic_process(x0)`` and simulates price paths.  From
    prices derives market weights and rank assignments at each step.

    Parameters
    ----------
    model : MarketModel
        The market model to simulate.
    x0 : ndarray of shape (n_assets,)
        Initial capitalizations (must be positive).
    T : float
        Terminal time.
    n_steps : int
        Number of time steps.
    seed : int, optional
        Random seed for reproducibility.
    discretization : Discretization, optional
        Numerical scheme.  Defaults to exact (``process.evolve``).
    track_local_times : bool
        If ``True``, estimate local time accumulation at adjacent
        rank boundaries using a Tanaka-formula discretisation.
    local_time_epsilon : float, optional
        Proximity threshold for local time estimation.  If ``None``,
        set adaptively from the data.

    Returns
    -------
    SPTResult[MarketSimulation]
        Simulation result with prices, weights, ranks, and optional
        local times.

    References
    ----------
    F&K Survey Eq. 1.2 (weights), Eq. 1.18 (ranks), BFK §3 (local times)
    """
    from ..models.base import MarketModel as MarketModelABC

    require(isinstance(model, MarketModelABC), "model must be a MarketModel")
    assert isinstance(model, MarketModelABC)
    require(T > 0, f"T must be positive, got {T}")
    require(n_steps > 0, f"n_steps must be positive, got {n_steps}")
    require(bool(np.all(x0 > 0)), "Initial values must be positive")

    with timed_result() as timer:
        process = model.to_stochastic_process(x0)
        rng = np.random.default_rng(seed)

        times, path = simulate_path(
            process,
            T=T,
            n_steps=n_steps,
            rng=rng,
            discretization=discretization,
        )

        # Convert log-space paths to prices if needed
        if hasattr(model, "log_space_process") and model.log_space_process:
            prices = np.exp(path)
        else:
            prices = path

        total_caps = prices.sum(axis=1, keepdims=True)
        weights = prices / total_caps

        n_assets = prices.shape[1]
        n_time = prices.shape[0]
        ranks = np.empty((n_time, n_assets), dtype=np.intp)
        for t_idx in range(n_time):
            order = np.argsort(-weights[t_idx])
            ranks[t_idx, order] = np.arange(n_assets)

        local_times_arr: NDArray[np.float64] | None = None
        if track_local_times and n_assets >= 2:
            local_times_arr = _estimate_local_times(prices, times, local_time_epsilon)

    sim = MarketSimulation(
        times=times,
        prices=prices,
        weights=weights,
        ranks=ranks,
        local_times=local_times_arr,
    )

    return SPTResult(
        data=sim,
        metadata={
            "model": type(model).__name__,
            "n_assets": n_assets,
            "T": T,
            "n_steps": n_steps,
            "track_local_times": track_local_times,
        },
        computation_time_ms=timer.elapsed_ms,
    )


def _estimate_local_times(
    prices: NDArray[np.float64],
    times: NDArray[np.float64],
    epsilon: float | None,
) -> NDArray[np.float64]:
    """Estimate cumulative local times from price paths.

    Uses a Tanaka-formula discretisation on log-prices.

    References
    ----------
    BFK §3
    """
    log_prices = np.log(prices)
    n_time, n_assets = log_prices.shape
    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0

    sorted_log = np.sort(log_prices, axis=1)[:, ::-1]
    gaps = -np.diff(sorted_log, axis=1)

    if epsilon is None:
        gap_std = float(np.std(gaps))
        epsilon = max(gap_std * 0.1, 1e-10)

    cumulative_lt = np.zeros((n_time, n_assets - 1))
    running = np.zeros(n_assets - 1)

    for t_idx in range(n_time):
        for k in range(n_assets - 1):
            if gaps[t_idx, k] < epsilon:
                running[k] += dt / epsilon
        cumulative_lt[t_idx] = running.copy()

    return cumulative_lt
