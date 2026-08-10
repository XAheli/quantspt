"""Simplex projection utilities for constrained portfolio construction.

Implements Euclidean projection onto the bounded simplex:
    {w ∈ ℝⁿ : w_i ≥ 0, Σ w_i = 1, w_i ≤ w_max}

The algorithm is based on sorting and threshold-finding, achieving O(n log n)
complexity. Numerically stable for large portfolios (n > 1000).

References
----------
- Duchi, Shalev-Shwartz, Singer, Chandra (2008): "Efficient Projections
  onto the ℓ₁-Ball for Learning in High Dimensions"
- Condat (2016): "Fast Projection onto the Simplex and the ℓ₁ Ball"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["project_bounded_simplex", "project_simplex"]


def project_simplex(v: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Project vector v onto the unit simplex Δ = {w ≥ 0, Σw = 1}.

    Finds w* = argmin ||w - v||² subject to w ∈ Δ.

    Parameters
    ----------
    v : ndarray of shape (n,)
        Input vector (unconstrained).

    Returns
    -------
    ndarray of shape (n,)
        Projected vector on the simplex.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u - (cssv - 1.0) / np.arange(1, n + 1)
    rho = int(np.max(np.where(rho_candidates > 0)[0])) + 1
    theta = (cssv[rho - 1] - 1.0) / rho
    return np.maximum(v - theta, 0.0)


def project_bounded_simplex(
    v: NDArray[np.float64],
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    budget: float = 1.0,
) -> NDArray[np.float64]:
    r"""Project vector onto bounded simplex {w : min ≤ w_i ≤ max, Σw = budget}.

    Uses iterative clipping with bisection on the Lagrange multiplier.
    Converges in O(n log n · log(1/ε)) for tolerance ε.

    Parameters
    ----------
    v : ndarray of shape (n,)
        Input vector (unconstrained target weights).
    max_weight : float
        Upper bound on each weight. Must be > 0 and ≤ 1.
    min_weight : float
        Lower bound on each weight. Must be ≥ 0 and < max_weight.
    budget : float
        Target sum of weights (default 1.0 for fully invested).

    Returns
    -------
    ndarray of shape (n,)
        Projected weights satisfying all constraints.

    Raises
    ------
    ValueError
        If constraints are infeasible (n * max_weight < budget or
        n * min_weight > budget).
    """
    n = len(v)
    if n * max_weight < budget - 1e-10:
        raise ValueError(
            f"Infeasible: {n} assets x max_weight={max_weight} = {n * max_weight:.4f} "
            f"< budget={budget}"
        )
    if n * min_weight > budget + 1e-10:
        raise ValueError(
            f"Infeasible: {n} assets x min_weight={min_weight} = {n * min_weight:.4f} "
            f"> budget={budget}"
        )

    if max_weight >= 1.0 and min_weight <= 0.0:
        return project_simplex(v) * budget

    lo = float(np.min(v)) - budget
    hi = float(np.max(v))

    for _ in range(100):
        mid = (lo + hi) / 2.0
        w = np.clip(v - mid, min_weight, max_weight)
        s = float(np.sum(w))
        if s > budget + 1e-12:
            lo = mid
        elif s < budget - 1e-12:
            hi = mid
        else:
            break

    w = np.clip(v - (lo + hi) / 2.0, min_weight, max_weight)
    residual = float(np.sum(w)) - budget
    if abs(residual) > 1e-10:
        active = (w > min_weight + 1e-12) & (w < max_weight - 1e-12)
        n_active = int(np.sum(active))
        if n_active > 0:
            w[active] -= residual / n_active

    return w
