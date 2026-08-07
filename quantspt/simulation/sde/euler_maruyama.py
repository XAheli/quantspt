"""Euler-Maruyama discretisation scheme with convergence utilities.

Re-exports the core Euler-Maruyama discretisation and adds convergence
order verification and adaptive step-size selection.

Mathematical References
-----------------------
- Euler-Maruyama scheme: Kloeden & Platen (1992), §9.1
- Strong convergence order 0.5
- Weak convergence order 1.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..._preconditions import require
from ...core.processes import (
    EulerMaruyamaDiscretization,
    simulate_path,
)

__all__ = [
    "EulerMaruyamaDiscretization",
    "adaptive_euler_maruyama",
    "verify_convergence_order",
]


def verify_convergence_order(
    process: object,
    T: float,
    step_counts: list[int],
    n_paths: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    r"""Verify strong convergence order of Euler-Maruyama vs exact solution.

    Computes the mean strong error for each step count and estimates
    the convergence rate from the log-log slope.  Euler-Maruyama has
    theoretical strong order 0.5, so the slope should be approximately
    −0.5.

    Parameters
    ----------
    process : StochasticProcess
        Must implement ``evolve`` for exact reference solutions.
    T : float
        Terminal time.
    step_counts : list of int
        Number of time steps to test (at least 2).
    n_paths : int
        Number of Monte Carlo paths per step count.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``'step_sizes'``: array of dt values,
        ``'errors'``: array of mean strong errors,
        ``'estimated_order'``: float (log-log slope, should be ≈ 0.5).

    References
    ----------
    Kloeden & Platen (1992), §9.1
    """
    require(len(step_counts) >= 2, "Need at least 2 step counts")
    require(T > 0, f"T must be positive, got {T}")
    require(n_paths > 0, f"n_paths must be positive, got {n_paths}")

    from ..._typing import StochasticProcess

    assert isinstance(process, StochasticProcess)

    em = EulerMaruyamaDiscretization()
    step_sizes = np.array([T / n for n in step_counts])
    errors = np.zeros(len(step_counts))

    for idx, n_steps in enumerate(step_counts):
        path_errors = np.zeros(n_paths)
        for p in range(n_paths):
            rng_em = np.random.default_rng(seed + p)
            rng_exact = np.random.default_rng(seed + p)

            _, path_em = simulate_path(
                process, T=T, n_steps=n_steps, rng=rng_em, discretization=em
            )
            _, path_exact = simulate_path(process, T=T, n_steps=n_steps, rng=rng_exact)
            path_errors[p] = float(np.max(np.abs(path_em[-1] - path_exact[-1])))

        errors[idx] = float(np.mean(path_errors))

    log_dt = np.log(step_sizes)
    log_err = np.log(errors + 1e-300)
    slope = float(np.polyfit(log_dt, log_err, 1)[0])

    return {
        "step_sizes": step_sizes,
        "errors": errors,
        "estimated_order": slope,
    }


def adaptive_euler_maruyama(
    process: object,
    T: float,
    dt_init: float,
    rng: np.random.Generator,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    dt_min: float = 1e-6,
    dt_max: float | None = None,
    max_steps: int = 500_000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Euler-Maruyama with adaptive step-size control.

    Uses a step-doubling error estimator: each step is computed at dt and
    at dt/2 (two half-steps).  The local error estimate drives step
    acceptance/rejection.

    The Brownian increment dW ~ N(0, dt·I) for the full step is split
    into two half-interval increments via a Brownian bridge conditioned
    on the total increment:

    .. math::

        Z &\sim N(0, I) \\
        dW_1 &= dW/2 + Z \sqrt{dt}/2 \\
        dW_2 &= dW/2 - Z \sqrt{dt}/2

    This ensures the correct statistical properties:

    - Partition: dW₁ + dW₂ = dW  (exact)
    - Var(dW₁) = Var(dW₂) = dt/2  (correct for half-interval)

    See Gaines & Lyons (1997) and Lamba et al. (2007) for adaptive
    SDE step-size control with consistent Brownian increments.

    The acceptance criterion uses mixed tolerance
    ``atol + rtol * max(|x|)`` so the step size scales with the
    solution magnitude (essential for GBM-like processes where
    drift and diffusion are proportional to *x*).

    Parameters
    ----------
    process : StochasticProcess
        The SDE to simulate.
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
        Minimum step size (prevents infinite refinement).
    dt_max : float, optional
        Maximum step size.  Defaults to ``dt_init * 4``.
    max_steps : int
        Safety limit on total accepted steps to prevent hangs.

    Returns
    -------
    tuple of (times, path)
        Times and path arrays with variable length.

    References
    ----------
    Kloeden & Platen (1992), §9.1 (adaptive extension)
    """
    from ..._typing import StochasticProcess

    assert isinstance(process, StochasticProcess)
    require(T > 0, f"T must be positive, got {T}")
    require(dt_init > 0, f"dt_init must be positive, got {dt_init}")
    require(atol > 0, f"atol must be positive, got {atol}")

    if dt_max is None:
        dt_max = dt_init * 4.0

    em = EulerMaruyamaDiscretization()
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

        x_full = em.evolve(process, t, x, dt, dw)

        z = rng.standard_normal(n_factors) * sqrt_dt * 0.5
        dw1 = dw * 0.5 + z
        dw2 = dw * 0.5 - z
        x_half = em.evolve(process, t, x, dt / 2.0, dw1)
        x_double = em.evolve(process, t + dt / 2.0, x_half, dt / 2.0, dw2)

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

    times_arr: NDArray[np.float64] = np.array(times_list)
    path_arr: NDArray[np.float64] = np.array(path_list)
    return times_arr, path_arr
