"""Monte Carlo simulation engine with variance reduction.

Provides a composable Monte Carlo framework following the
PathGenerator → PathPricer → Accumulator architecture.  Supports
antithetic variates and Brownian bridge path construction for
variance reduction.

Mathematical References
-----------------------
- Monte Carlo with antithetic variates: standard variance reduction technique
- GBM analytical mean: E[S(T)] = S(0) exp(μT)
- Confidence intervals via CLT
- Brownian bridge construction: Jäckel (2002), Ch. 10
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from .._result import SPTResult, timed_result
from ..core.processes import (
    simulate_path,
)
from .brownian_bridge import BrownianBridge

__all__ = [
    "MonteCarloEngine",
    "MonteCarloResult",
]


@dataclass
class MonteCarloResult:
    """Structured result from Monte Carlo simulation.

    Attributes
    ----------
    terminal_values : ndarray of shape (n_effective_paths, n_assets)
        Terminal state of each path.
    mean : ndarray of shape (n_assets,)
        Sample mean across paths.
    std : ndarray of shape (n_assets,)
        Sample standard deviation across paths.
    ci_lower : ndarray of shape (n_assets,)
        Lower bound of confidence interval.
    ci_upper : ndarray of shape (n_assets,)
        Upper bound of confidence interval.
    confidence_level : float
        Confidence level used for intervals.
    n_paths : int
        Number of effective paths (accounting for antithetic doubling).
    antithetic : bool
        Whether antithetic variates were used.
    """

    terminal_values: NDArray[np.float64]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    confidence_level: float
    n_paths: int
    antithetic: bool


class MonteCarloEngine:
    r"""Monte Carlo engine for stochastic process simulation.

    Runs *n_paths* independent simulations of a ``StochasticProcess``
    and collects terminal statistics including mean, standard deviation,
    and confidence intervals.

    Supports **antithetic variates** for variance reduction: each path
    is paired with its antithetic counterpart (negated Brownian increments),
    and the estimator uses the average of each pair.

    When ``bridge=True``, Brownian increments are generated via
    binary-tree bridge construction (Jäckel 2002, Ch. 10) instead of
    sequential cumulative sampling.  This reorders the variates so that
    the terminal value and coarse-scale features consume the first
    random coordinates — enabling better quasi-random coupling with
    Sobol sequences.

    Parameters
    ----------
    process : StochasticProcess
        The stochastic process to simulate.
    n_paths : int
        Number of Monte Carlo paths.
    T : float
        Terminal time.
    n_steps : int
        Number of discretisation steps.
    discretization : Discretization, optional
        Numerical scheme.  Defaults to exact (``process.evolve``).
    antithetic : bool
        Whether to use antithetic variates for variance reduction.
    bridge : bool
        Whether to use Brownian bridge path construction.
    confidence_level : float
        Confidence level for intervals (default 0.95).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        process: object,
        n_paths: int = 10_000,
        T: float = 1.0,
        n_steps: int = 252,
        discretization: object | None = None,
        antithetic: bool = False,
        bridge: bool = False,
        confidence_level: float = 0.95,
        seed: int | None = None,
    ) -> None:
        from .._typing import StochasticProcess

        assert isinstance(process, StochasticProcess)

        require(n_paths > 0, f"n_paths must be positive, got {n_paths}")
        require(T > 0, f"T must be positive, got {T}")
        require(n_steps > 0, f"n_steps must be positive, got {n_steps}")
        require(
            0 < confidence_level < 1,
            f"confidence_level must be in (0, 1), got {confidence_level}",
        )

        self._process = process
        self._n_paths = n_paths
        self._T = T
        self._n_steps = n_steps
        self._discretization = discretization
        self._antithetic = antithetic
        self._bridge = bridge
        self._confidence_level = confidence_level
        self._rng = np.random.default_rng(seed)

        self._bb: BrownianBridge | None
        if bridge:
            times = np.linspace(0.0, T, n_steps + 1).astype(np.float64)
            self._bb = BrownianBridge(times)
        else:
            self._bb = None

    def run(self) -> SPTResult[MonteCarloResult]:
        """Execute the Monte Carlo simulation.

        Returns
        -------
        SPTResult[MonteCarloResult]
            Result envelope containing terminal statistics, CI, and metadata.
        """
        with timed_result() as timer:
            if self._antithetic:
                terminal_values = self._run_antithetic()
            else:
                terminal_values = self._run_plain()

            mean = np.mean(terminal_values, axis=0)
            std = np.std(terminal_values, axis=0, ddof=1)
            n_eff = terminal_values.shape[0]

            from scipy import stats

            alpha = 1.0 - self._confidence_level
            z = stats.norm.ppf(1.0 - alpha / 2.0)
            se = std / np.sqrt(n_eff)
            ci_lower = mean - z * se
            ci_upper = mean + z * se

        mc_result = MonteCarloResult(
            terminal_values=terminal_values,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self._confidence_level,
            n_paths=n_eff,
            antithetic=self._antithetic,
        )

        return SPTResult(
            data=mc_result,
            metadata={
                "n_paths_requested": self._n_paths,
                "n_effective_paths": n_eff,
                "T": self._T,
                "n_steps": self._n_steps,
                "antithetic": self._antithetic,
                "bridge": self._bridge,
                "confidence_level": self._confidence_level,
            },
            computation_time_ms=timer.elapsed_ms,
        )

    def _run_plain(self) -> NDArray[np.float64]:
        """Run plain Monte Carlo without variance reduction."""
        n_size = self._process.size()
        terminals = np.empty((self._n_paths, n_size))

        if self._bridge:
            n_factors = self._process.factors()
            dt = self._T / self._n_steps
            for i in range(self._n_paths):
                path_rng = np.random.default_rng(self._rng.integers(0, 2**63))
                terminals[i] = self._simulate_with_bridge(path_rng, n_factors, dt)
        else:
            for i in range(self._n_paths):
                path_rng = np.random.default_rng(self._rng.integers(0, 2**63))
                _, path = simulate_path(
                    self._process,
                    T=self._T,
                    n_steps=self._n_steps,
                    rng=path_rng,
                    discretization=self._discretization,
                )
                terminals[i] = path[-1]

        return terminals

    def _run_antithetic(self) -> NDArray[np.float64]:
        """Run with antithetic variates.

        For each base path with Brownian increments dW, the antithetic
        path uses -dW.  The estimator is the average of each pair,
        which reduces variance when the payoff is monotone in the noise.
        """
        n_size = self._process.size()
        n_factors = self._process.factors()
        dt = self._T / self._n_steps
        sqrt_dt = np.sqrt(dt)

        pair_means = np.empty((self._n_paths, n_size))

        for i in range(self._n_paths):
            path_seed = self._rng.integers(0, 2**63)

            if self._bridge:
                terminal_base = self._simulate_with_bridge(
                    np.random.default_rng(int(path_seed)), n_factors, dt, negate=False
                )
                terminal_anti = self._simulate_with_bridge(
                    np.random.default_rng(int(path_seed)), n_factors, dt, negate=True
                )
            else:
                terminal_base = self._simulate_with_noise(
                    int(path_seed), n_factors, dt, sqrt_dt, negate=False
                )
                terminal_anti = self._simulate_with_noise(
                    int(path_seed), n_factors, dt, sqrt_dt, negate=True
                )

            pair_means[i] = 0.5 * (terminal_base + terminal_anti)

        return pair_means

    def _simulate_with_noise(
        self,
        seed: int,
        n_factors: int,
        dt: float,
        sqrt_dt: float,
        negate: bool,
    ) -> NDArray[np.float64]:
        """Simulate a single path with optionally negated noise."""
        from .._typing import Discretization

        rng = np.random.default_rng(seed)
        x = self._process.initial_values()

        for k in range(self._n_steps):
            dw = rng.standard_normal(n_factors) * sqrt_dt
            if negate:
                dw = -dw

            t_k = k * dt
            if self._discretization is not None:
                assert isinstance(self._discretization, Discretization)
                x = self._discretization.evolve(self._process, t_k, x, dt, dw)
            else:
                x = self._process.evolve(t_k, x, dt, dw)

        return x

    def _simulate_with_bridge(
        self,
        rng: np.random.Generator,
        n_factors: int,
        dt: float,
        negate: bool = False,
    ) -> NDArray[np.float64]:
        """Simulate a single path using Brownian bridge construction."""
        from .._typing import Discretization

        assert self._bb is not None
        normals = rng.standard_normal((self._n_steps, n_factors))
        if negate:
            normals = -normals

        increments = self._bb.increments(normals)

        x = self._process.initial_values()
        for k in range(self._n_steps):
            dw = increments[k]
            t_k = k * dt
            if self._discretization is not None:
                assert isinstance(self._discretization, Discretization)
                x = self._discretization.evolve(self._process, t_k, x, dt, dw)
            else:
                x = self._process.evolve(t_k, x, dt, dw)

        return x
