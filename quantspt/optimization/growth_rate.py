r"""Growth rate maximisation -- the correct SPT objective.

Maximises the portfolio growth rate:

.. math::
    \gamma_{\pi} = \sum_i \pi_i \gamma_i
                   + \frac{1}{2}\left[
                       \sum_i \pi_i a_{ii} - \pi^T a \pi
                   \right]

This is a concave maximisation (convex minimisation) since the excess
growth rate :math:`-\pi^T a \pi` term is concave for PSD a.

IMPORTANT: This is NOT Markowitz mean-variance. The SPT objective
naturally balances return and diversification benefit in a single
quantity -- the portfolio growth rate.

Mathematical References
-----------------------
- Portfolio growth rate: F&K Survey Eq. 1.12-1.13
- Excess growth rate: F&K Survey Eq. 1.13, FKK Eq. 2.8
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from ..errors import InfeasibleError, OptimizationError, SPTInvariantError

__all__ = [
    "optimize_growth_rate",
]


def optimize_growth_rate(
    growth_rates: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
    *,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    max_turnover: float | None = None,
    prev_weights: NDArray[np.float64] | None = None,
    max_tracking_error: float | None = None,
    benchmark: NDArray[np.float64] | None = None,
    extra_constraints: list[object] | None = None,
    solver: str = "SCS",
) -> dict[str, NDArray[np.float64] | float | str]:
    r"""Maximise portfolio growth rate (the correct SPT objective).

    Solves:

    .. math::
        \max_{\pi} \quad \gamma_{\pi} = \sum_i \pi_i \gamma_i
            + \frac{1}{2}\left[\sum_i \pi_i a_{ii} - \pi^T a \pi\right]

    subject to:

    .. math::
        \sum_i \pi_i = 1, \quad
        \pi_i \in [\text{min\_weight}, \text{max\_weight}]

    and optional turnover and tracking error constraints.

    Parameters
    ----------
    growth_rates : ndarray of shape (n,)
        Individual stock growth rates gamma_i.
    cov_matrix : ndarray of shape (n, n)
        Annualised covariance rate matrix (must be PSD).
    min_weight : float
        Minimum weight per asset (default 0 = long only).
    max_weight : float
        Maximum weight per asset (default 1 = no cap).
    max_turnover : float or None
        Maximum one-way turnover. Requires *prev_weights*.
    prev_weights : ndarray of shape (n,) or None
        Previous portfolio weights (for turnover constraint).
    max_tracking_error : float or None
        Maximum tracking error vs *benchmark*.
    benchmark : ndarray of shape (n,) or None
        Benchmark weights (for tracking error constraint).
    extra_constraints : list or None
        Additional CVXPY constraint objects.
    solver : str
        Preferred CVXPY solver (default ``'SCS'``).

    Returns
    -------
    dict with keys:
        ``'weights'`` : ndarray of shape (n,)
            Optimal portfolio weights.
        ``'growth_rate'`` : float
            Achieved portfolio growth rate.
        ``'excess_growth_rate'`` : float
            Excess growth rate component.
        ``'status'`` : str
            Solver status.

    Raises
    ------
    OptimizationError
        If all solvers fail.
    InfeasibleError
        If the problem has no feasible solution.

    References
    ----------
    F&K Survey Eq. 1.12-1.13
    """
    import cvxpy as cp

    growth_rates = np.asarray(growth_rates, dtype=np.float64)
    cov_matrix = np.asarray(cov_matrix, dtype=np.float64)

    n = len(growth_rates)
    require(
        cov_matrix.shape == (n, n),
        f"Covariance shape {cov_matrix.shape} incompatible with {n} assets",
    )
    require(min_weight <= max_weight, "min_weight must be <= max_weight")

    # Enforce PSD property on covariance matrix
    cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
    min_eig = float(np.min(np.linalg.eigvalsh(cov_matrix)))
    if min_eig < -1e-6:
        raise SPTInvariantError(
            f"Covariance matrix is not PSD: smallest eigenvalue = {min_eig:.2e}. "
            "Consider using a shrinkage estimator (ledoit_wolf)."
        )
    if min_eig < 0:
        cov_matrix = cov_matrix - min_eig * np.eye(n)

    pi = cp.Variable(n)

    weighted_growth = growth_rates @ pi
    weighted_var = np.diag(cov_matrix) @ pi
    port_var = cp.quad_form(pi, cov_matrix)  # type: ignore[attr-defined]

    objective = cp.Maximize(weighted_growth + 0.5 * (weighted_var - port_var))

    constraints: list[object] = [
        cp.sum(pi) == 1,  # type: ignore[attr-defined]
        pi >= min_weight,
        pi <= max_weight,
    ]

    if max_turnover is not None and prev_weights is not None:
        prev_weights = np.asarray(prev_weights, dtype=np.float64)
        constraints.append(cp.norm1(pi - prev_weights) <= 2 * max_turnover)  # type: ignore[attr-defined]

    if max_tracking_error is not None and benchmark is not None:
        benchmark = np.asarray(benchmark, dtype=np.float64)
        tracking = cp.quad_form(pi - benchmark, cov_matrix)  # type: ignore[attr-defined]
        constraints.append(tracking <= max_tracking_error**2)

    if extra_constraints:
        for c in extra_constraints:
            constraints.append(c)

    problem = cp.Problem(objective, constraints)  # type: ignore[arg-type]

    solver_chain = list(dict.fromkeys([solver, "SCS", "ECOS", "OSQP"]))
    last_status = ""
    for s in solver_chain:
        try:
            problem.solve(solver=s)  # type: ignore[no-untyped-call]
            last_status = str(problem.status)
            if problem.status in ("optimal", "optimal_inaccurate"):
                break
        except cp.SolverError:
            continue

    if problem.status == "infeasible":
        raise InfeasibleError(
            f"No feasible solution: min_weight={min_weight}, max_weight={max_weight}"
        )

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise OptimizationError(
            f"All solvers failed. Last status: {last_status}. Tried: {solver_chain}"
        )

    w_arr = np.asarray(pi.value, dtype=np.float64).flatten()
    w = np.clip(w_arr, min_weight, max_weight)
    w /= w.sum()

    weighted_g = float(np.dot(w, growth_rates))
    excess_g = 0.5 * (float(np.dot(w, np.diag(cov_matrix))) - float(w @ cov_matrix @ w))

    return {
        "weights": w,
        "growth_rate": weighted_g + excess_g,
        "excess_growth_rate": excess_g,
        "status": str(problem.status),
    }
