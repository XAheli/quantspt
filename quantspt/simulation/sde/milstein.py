"""Milstein discretisation scheme with convergence utilities.

Re-exports the core Milstein discretisation and adds convergence
order verification and adaptive step-size selection for scalar SDEs.

Mathematical References
-----------------------
- Milstein scheme: Kloeden & Platen (1992), Theorem 10.3.5
- Strong convergence order 1.0 (for scalar SDEs)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..._preconditions import require
from ...core.processes import (
    MilsteinDiscretization,
    simulate_path,
)

__all__ = [
    "MilsteinDiscretization",
    "adaptive_milstein",
    "verify_milstein_convergence",
]


def verify_milstein_convergence(
    process: object,
    T: float,
    step_counts: list[int],
    n_paths: int = 200,
    seed: int = 42,
    diffusion_deriv: object | None = None,
) -> dict[str, object]:
    r"""Verify strong convergence order of Milstein vs exact solution.

    For scalar SDEs, Milstein has theoretical strong order 1.0,
    so the log-log slope of error vs dt should be approximately 1.0.

    Parameters
    ----------
    process : StochasticProcess
        1-D process with ``evolve`` for exact reference solutions.
    T : float
        Terminal time.
    step_counts : list of int
        Number of time steps to test (at least 2).
    n_paths : int
        Number of Monte Carlo paths per step count.
    seed : int
        Random seed for reproducibility.
    diffusion_deriv : callable, optional
        Analytical derivative dσ/dx for higher accuracy.

    Returns
    -------
    dict
        ``'step_sizes'``: array of dt values,
        ``'errors'``: array of mean strong errors,
        ``'estimated_order'``: float (should be ≈ 1.0 for scalar SDEs).

    References
    ----------
    Kloeden & Platen (1992), Theorem 10.3.5
    """
    require(len(step_counts) >= 2, "Need at least 2 step counts")
    require(T > 0, f"T must be positive, got {T}")
    require(n_paths > 0, f"n_paths must be positive, got {n_paths}")

    from ..._typing import StochasticProcess

    assert isinstance(process, StochasticProcess)
    require(process.size() == 1, "Milstein convergence test requires 1-D process")

    mil = MilsteinDiscretization(diffusion_deriv=diffusion_deriv)
    step_sizes = np.array([T / n for n in step_counts])
    errors = np.zeros(len(step_counts))

    for idx, n_steps in enumerate(step_counts):
        path_errors = np.zeros(n_paths)
        for p in range(n_paths):
            rng_mil = np.random.default_rng(seed + p)
            rng_exact = np.random.default_rng(seed + p)

            _, path_mil = simulate_path(
                process, T=T, n_steps=n_steps, rng=rng_mil, discretization=mil
            )
            _, path_exact = simulate_path(process, T=T, n_steps=n_steps, rng=rng_exact)
            path_errors[p] = float(np.abs(path_mil[-1, 0] - path_exact[-1, 0]))

        errors[idx] = float(np.mean(path_errors))

    log_dt = np.log(step_sizes)
    log_err = np.log(errors + 1e-300)
    slope = float(np.polyfit(log_dt, log_err, 1)[0])

    return {
        "step_sizes": step_sizes,
        "errors": errors,
        "estimated_order": slope,
    }


def adaptive_milstein(
    process: object,
    T: float,
    dt_init: float,
    rng: np.random.Generator,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    dt_min: float = 1e-6,
    dt_max: float | None = None,
    max_steps: int = 500_000,
    diffusion_deriv: object | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Milstein scheme with adaptive step-size control (1-D only).

    Uses step-doubling error estimation with the Milstein correction
    for improved local accuracy.

    The acceptance criterion uses mixed tolerance
    ``atol + rtol * max(|x|)`` so the step size scales with the
    solution magnitude.

    Parameters
    ----------
    process : StochasticProcess
        1-D SDE to simulate.
    T : float
        Terminal time.
    dt_init : float
        Initial step size.
    rng : numpy.random.Generator
        Random number generator.
    atol : float
        Absolute error tolerance component.
    rtol : float
        Relative error tolerance component, scaled by ``max(|x|)``.
    dt_min : float
        Minimum step size.
    dt_max : float, optional
        Maximum step size.  Defaults to ``dt_init * 4``.
    max_steps : int
        Safety limit on total accepted steps to prevent hangs.
    diffusion_deriv : callable, optional
        Analytical dσ/dx.

    Returns
    -------
    tuple of (times, path)
        Times and path arrays with variable length.

    References
    ----------
    Kloeden & Platen (1992), Theorem 10.3.5
    """
    from ..._typing import StochasticProcess

    assert isinstance(process, StochasticProcess)
    require(process.size() == 1, "Adaptive Milstein requires 1-D process")
    require(T > 0, f"T must be positive, got {T}")
    require(dt_init > 0, f"dt_init must be positive, got {dt_init}")

    if dt_max is None:
        dt_max = dt_init * 4.0

    mil = MilsteinDiscretization(diffusion_deriv=diffusion_deriv)
    n_factors = process.factors()

    times_list: list[float] = [0.0]
    path_list: list[NDArray[np.float64]] = [process.initial_values()]

    t = 0.0
    x = process.initial_values()
    dt = dt_init
    n_accepted = 0

    while t < T - 1e-14:
        dt = min(dt, T - t)
        dt = max(dt, dt_min)
        sqrt_dt = np.sqrt(dt)

        dw = rng.standard_normal(n_factors) * sqrt_dt
        x_full = mil.evolve(process, t, x, dt, dw)

        dw1 = dw * np.sqrt(0.5)
        dw2 = dw * np.sqrt(0.5)
        x_half = mil.evolve(process, t, x, dt / 2.0, dw1)
        x_double = mil.evolve(process, t + dt / 2.0, x_half, dt / 2.0, dw2)

        error = float(np.max(np.abs(x_full - x_double)))
        tol = atol + rtol * float(np.max(np.abs(x)))

        if error < tol or dt <= dt_min:
            x = x_double
            t += dt
            times_list.append(t)
            path_list.append(x.copy())
            n_accepted += 1

            if n_accepted >= max_steps:
                break

            if error < tol * 0.1 and dt < dt_max:
                dt = min(dt * 2.0, dt_max)
        else:
            dt = max(dt * 0.5, dt_min)

    return np.array(times_list), np.array(path_list)
