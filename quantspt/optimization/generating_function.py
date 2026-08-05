r"""Generating function parameter optimisation.

Finds the optimal parameters for a generating function (e.g., the best p
for DiversityGenerator) by maximising an objective such as the expected
drift integral, Sharpe ratio of relative return, or growth rate.

Supports grid search over a parameter range and refinement via
scipy.optimize for continuous parameters.

Mathematical References
-----------------------
- FGP drift process: F&K Survey Eq. 11.3
- Diversity generator G_p: F&K Survey Remark 11.1, FKK Eq. 4.4
- Master formula: F&K Survey Eq. 11.2
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from .._preconditions import require
from ..core.covariance import relative_covariance
from ..core.generating_functions import DiversityGenerator, GeneratingFunction

__all__ = [
    "OptimizationResult",
    "optimize_diversity_parameter",
    "optimize_generator_parameter",
]


@dataclass(frozen=True)
class OptimizationResult:
    """Result of generating function parameter optimisation.

    Attributes
    ----------
    optimal_param : float
        Best parameter value found.
    optimal_value : float
        Objective function value at the optimum.
    grid_params : ndarray
        Parameter values evaluated in the grid search.
    grid_values : ndarray
        Objective values at each grid point.
    method : str
        Optimisation method used.
    """

    optimal_param: float
    optimal_value: float
    grid_params: NDArray[np.float64]
    grid_values: NDArray[np.float64]
    method: str


def _mean_drift(
    generator_factory: Callable[[float], GeneratingFunction],
    param: float,
    weights: NDArray[np.float64],
    cov_matrices: NDArray[np.float64] | list[NDArray[np.float64]],
) -> float:
    """Compute mean drift of a generating function over a time series.

    Parameters
    ----------
    generator_factory : callable
        Maps parameter value to a GeneratingFunction instance.
    param : float
        Parameter value to evaluate.
    weights : ndarray of shape (T, n)
        Time series of market weights.
    cov_matrices : ndarray of shape (T, n, n) or list of (n, n) arrays
        Covariance matrices at each time step.

    Returns
    -------
    float
        Average drift process value.
    """
    G = generator_factory(param)
    T = weights.shape[0]
    total_drift = 0.0

    for t in range(T):
        mu_t = weights[t]
        if isinstance(cov_matrices, list):
            cov_t = cov_matrices[t]
        else:
            cov_t = cov_matrices[t]
        tau_t = relative_covariance(cov_t, mu_t)
        total_drift += G.drift(mu_t, tau_t)

    return total_drift / T


def _sharpe_of_relative_return(
    generator_factory: Callable[[float], GeneratingFunction],
    param: float,
    weights: NDArray[np.float64],
    cov_matrices: NDArray[np.float64] | list[NDArray[np.float64]],
) -> float:
    """Compute Sharpe ratio of the drift process time series.

    A higher Sharpe indicates more consistent outperformance of the
    market portfolio.
    """
    G = generator_factory(param)
    T = weights.shape[0]
    drifts = np.empty(T)

    for t in range(T):
        mu_t = weights[t]
        if isinstance(cov_matrices, list):
            cov_t = cov_matrices[t]
        else:
            cov_t = cov_matrices[t]
        tau_t = relative_covariance(cov_t, mu_t)
        drifts[t] = G.drift(mu_t, tau_t)

    mean_d = float(np.mean(drifts))
    std_d = float(np.std(drifts, ddof=1))
    if std_d < 1e-15:
        return mean_d * 1e10 if mean_d > 0 else 0.0
    return mean_d / std_d


_OBJECTIVE_MAP: dict[
    str,
    Callable[
        [
            Callable[[float], GeneratingFunction],
            float,
            NDArray[np.float64],
            NDArray[np.float64] | list[NDArray[np.float64]],
        ],
        float,
    ],
] = {
    "mean_drift": _mean_drift,
    "sharpe": _sharpe_of_relative_return,
}


def optimize_generator_parameter(
    generator_factory: Callable[[float], GeneratingFunction],
    weights: NDArray[np.float64],
    cov_matrices: NDArray[np.float64] | list[NDArray[np.float64]],
    *,
    param_range: tuple[float, float] = (0.1, 0.9),
    n_grid: int = 50,
    refine: bool = True,
    objective: Literal["mean_drift", "sharpe"] = "mean_drift",
) -> OptimizationResult:
    r"""Optimise a scalar parameter of a generating function.

    Performs a grid search over the parameter range, then optionally
    refines with Brent's method (scipy.optimize.minimize_scalar).

    Parameters
    ----------
    generator_factory : callable
        Function that takes a float parameter and returns a
        :class:`~quantspt.core.generating_functions.GeneratingFunction`.
    weights : ndarray of shape (T, n)
        Time series of market weights.
    cov_matrices : ndarray of shape (T, n, n) or list of (n, n) arrays
        Covariance matrices at each time step.
    param_range : tuple of (float, float)
        (lower, upper) bounds for the parameter search.
    n_grid : int
        Number of grid points for initial search.
    refine : bool
        If ``True``, refine the grid optimum with Brent's method.
    objective : str
        Objective function: ``'mean_drift'`` or ``'sharpe'``.

    Returns
    -------
    OptimizationResult
        Contains optimal parameter, objective value, and grid data.

    References
    ----------
    F&K Survey Eq. 11.3 (drift process)
    """
    weights = np.asarray(weights, dtype=np.float64)
    require(weights.ndim == 2, f"weights must be 2-D, got shape {weights.shape}")
    require(
        param_range[0] < param_range[1],
        f"param_range must be (lo, hi), got {param_range}",
    )
    require(
        objective in _OBJECTIVE_MAP,
        f"Unknown objective '{objective}', choose from {list(_OBJECTIVE_MAP)}",
    )

    obj_fn = _OBJECTIVE_MAP[objective]

    grid_params = np.linspace(param_range[0], param_range[1], n_grid)
    grid_values = np.empty(n_grid)

    for i, p in enumerate(grid_params):
        try:
            grid_values[i] = obj_fn(generator_factory, p, weights, cov_matrices)
        except Exception:
            grid_values[i] = -np.inf

    best_idx = int(np.argmax(grid_values))
    best_param = float(grid_params[best_idx])
    best_value = float(grid_values[best_idx])

    method = "grid"

    if refine and n_grid > 2:

        def neg_objective(p: float) -> float:
            try:
                return -obj_fn(generator_factory, p, weights, cov_matrices)
            except Exception:
                return np.inf

        result = optimize.minimize_scalar(
            neg_objective,
            bounds=param_range,
            method="bounded",
        )
        if result.success and -result.fun > best_value:
            best_param = float(result.x)
            best_value = float(-result.fun)
            method = "grid+brent"

    return OptimizationResult(
        optimal_param=best_param,
        optimal_value=best_value,
        grid_params=np.asarray(grid_params, dtype=np.float64),
        grid_values=grid_values,
        method=method,
    )


def optimize_diversity_parameter(
    weights: NDArray[np.float64],
    cov_matrices: NDArray[np.float64] | list[NDArray[np.float64]],
    *,
    p_range: tuple[float, float] = (0.1, 0.9),
    n_grid: int = 50,
    refine: bool = True,
    objective: Literal["mean_drift", "sharpe"] = "mean_drift",
) -> OptimizationResult:
    r"""Find the optimal diversity parameter p for DiversityGenerator.

    Convenience wrapper around :func:`optimize_generator_parameter`
    for the most common use case.

    Parameters
    ----------
    weights : ndarray of shape (T, n)
        Time series of market weights.
    cov_matrices : ndarray of shape (T, n, n) or list of (n, n) arrays
        Covariance matrices at each time step.
    p_range : tuple of (float, float)
        Search range for p (must be within (0, 1)).
    n_grid : int
        Number of grid points.
    refine : bool
        Refine with Brent's method.
    objective : str
        ``'mean_drift'`` or ``'sharpe'``.

    Returns
    -------
    OptimizationResult
        Contains optimal p and objective value.

    References
    ----------
    F&K Survey Remark 11.1, FKK Eq. 4.4
    """
    require(
        0 < p_range[0] < p_range[1] < 1,
        f"p_range must be within (0, 1), got {p_range}",
    )

    def factory(p: float) -> GeneratingFunction:
        return DiversityGenerator(p)

    return optimize_generator_parameter(
        factory,
        weights,
        cov_matrices,
        param_range=p_range,
        n_grid=n_grid,
        refine=refine,
        objective=objective,
    )
