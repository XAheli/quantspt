"""Multivariate Hawkes process estimation and simulation.

Self-exciting point processes for modeling volatility clustering, trade
arrival clustering, and momentum cascades in financial markets. The key
insight: past events increase the probability of future events, captured
by the branching ratio n = α/β which measures market endogeneity
(n ≈ 0.7–0.85 for liquid equities).

Mathematical References
-----------------------
- Hawkes intensity: Hawkes (1971), "Spectra of some self-exciting and
  mutually exciting point processes," Biometrika 58(1), pp. 83-90.

  Multivariate intensity with exponential kernel:
    λ_i(t) = μ_i + Σ_j α_{ij} Σ_{t_jk < t} β_{ij} exp[−β_{ij}(t − t_jk)]

- Branching ratio and endogeneity: Bacry, Mastromatteo & Muzy (2015),
  "Hawkes processes in finance," Quantitative Finance 15(7), pp. 1147-1160.
  The spectral radius ρ(α) < 1 is the stationarity condition; when
  ρ → 1 the process approaches criticality.  (Since the kernel
  α_{ij}·β_{ij}·exp(−β_{ij}·t) integrates to α_{ij}, the branching
  matrix is simply α, not α/β.)

- Parameter estimation: Ozaki (1979), "Maximum likelihood estimation of
  Hawkes' self-exciting point processes," Ann. Inst. Statist. Math. 31,
  pp. 145-155. The log-likelihood admits a recursive O(N) computation.

- Simulation: Ogata (1981), "On Lewis' simulation method for point
  processes," IEEE Trans. Inform. Theory 27(1), pp. 23-31. Thinning
  algorithm with adaptive upper bound.

- Hardiman, Bercot & Bouchaud (2013), "Critical reflexivity in financial
  markets: a Hawkes process analysis," Eur. Phys. J. B 86, Art. 442.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .._preconditions import require
from .._result import SPTResult

__all__ = [
    "HawkesProcess",
    "HawkesResult",
    "simulate_hawkes",
]


@dataclass(frozen=True)
class HawkesResult:
    """Fitted Hawkes process parameters.

    Attributes
    ----------
    mu : NDArray[np.float64]
        Baseline intensities, shape ``(d,)``.
    alpha : NDArray[np.float64]
        Excitation magnitudes, shape ``(d, d)``.
    beta : NDArray[np.float64]
        Decay rates, shape ``(d, d)``.
    branching_ratio : float
        Spectral radius of α, measuring endogeneity (since
        ∫₀^∞ α·β·exp(−β·t) dt = α).
    log_likelihood : float
        Maximized log-likelihood value.
    n_events : list[int]
        Number of events per dimension.
    """

    mu: NDArray[np.float64]
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    branching_ratio: float
    log_likelihood: float
    n_events: list[int]


class HawkesProcess:
    r"""Multivariate Hawkes process with exponential kernels.

    The conditional intensity of dimension i is:

    .. math::
        \lambda_i(t) = \mu_i + \sum_{j=1}^{d} \alpha_{ij}
        \sum_{t_{jk} < t} \beta_{ij} \exp\bigl[-\beta_{ij}(t - t_{jk})\bigr]

    where μ_i > 0 is the baseline intensity, α_{ij} ≥ 0 is the excitation
    magnitude from dimension j to i, and β_{ij} > 0 is the decay rate.

    The branching ratio (spectral radius of the matrix α_{ij}) must be
    < 1 for stationarity.  The kernel α·β·exp(−β·t) integrates to α,
    so the branching matrix is α itself.  Values near 1 indicate
    critical reflexivity in the market (Hardiman et al., 2013).

    Parameters
    ----------
    n_dim : int
        Number of dimensions (event types).

    References
    ----------
    Bacry, Mastromatteo & Muzy (2015), "Hawkes processes in finance,"
    Quantitative Finance 15(7), pp. 1147-1160, Eq. (1)-(5).
    """

    def __init__(self, n_dim: int = 1) -> None:
        require(n_dim >= 1, f"n_dim must be >= 1, got {n_dim}")
        self._d = n_dim
        self._mu: NDArray[np.float64] | None = None
        self._alpha: NDArray[np.float64] | None = None
        self._beta: NDArray[np.float64] | None = None
        self._fitted = False

    def fit(
        self,
        events: list[NDArray[np.float64]],
        T: float,
        *,
        max_iter: int = 200,
        initial_params: dict[str, NDArray[np.float64]] | None = None,
    ) -> SPTResult[HawkesResult]:
        r"""Estimate parameters via maximum likelihood.

        Uses the log-likelihood computation (cf. Ozaki, 1979):

        .. math::
            \log L = \sum_i \Bigl[\sum_k \log \lambda_i(t_{ik})
                     - \int_0^T \lambda_i(s)\,ds\Bigr]

        The integral of the exponential kernel has closed form:

        .. math::
            \int_0^T \alpha\beta e^{-\beta(T-s)} ds
            = \alpha \bigl[1 - e^{-\beta T}\bigr]
            \quad\text{(per event contribution)}

        Parameters
        ----------
        events : list of ndarray
            ``events[i]`` is a sorted 1-D array of event times for
            dimension i. Length of list must equal ``n_dim``.
        T : float
            Observation window [0, T].
        max_iter : int
            Maximum L-BFGS-B iterations.
        initial_params : dict, optional
            Initial guesses with keys ``'mu'``, ``'alpha'``, ``'beta'``.

        Returns
        -------
        SPTResult[HawkesResult]
            Fitted parameters, branching ratio, and log-likelihood.

        References
        ----------
        Ozaki (1979), Ann. Inst. Statist. Math. 31, pp. 145-155.
        """
        t0 = time.perf_counter()
        d = self._d
        require(len(events) == d, f"Expected {d} event arrays, got {len(events)}")
        require(T > 0, f"T must be positive, got {T}")

        for i, ev in enumerate(events):
            require(ev.ndim == 1, f"events[{i}] must be 1-D")
            if len(ev) > 1:
                require(
                    bool(np.all(np.diff(ev) >= 0)),
                    f"events[{i}] must be sorted",
                )

        if initial_params is not None:
            mu0 = initial_params.get("mu", np.full(d, 0.1))
            alpha0 = initial_params.get("alpha", np.full((d, d), 0.05))
            beta0 = initial_params.get("beta", np.full((d, d), 1.0))
        else:
            total_events = sum(len(ev) for ev in events)
            avg_rate = total_events / (d * T) if T > 0 else 1.0
            mu0 = np.full(d, avg_rate * 0.5)
            alpha0 = np.full((d, d), 0.1)
            beta0 = np.full((d, d), 1.0)

        x0 = np.concatenate([mu0.ravel(), alpha0.ravel(), beta0.ravel()])
        bounds = (
            [(1e-8, None)] * d + [(1e-8, None)] * (d * d) + [(1e-4, None)] * (d * d)
        )

        def neg_ll(params: NDArray[np.float64]) -> float:
            mu, alpha, beta = self._unpack(params)
            return -self._log_likelihood(events, T, mu, alpha, beta)

        result = minimize(
            neg_ll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-10},
        )

        mu, alpha, beta = self._unpack(result.x)
        self._mu = mu
        self._alpha = alpha
        self._beta = beta
        self._fitted = True

        branching_matrix = alpha
        branching_ratio = float(np.max(np.abs(np.linalg.eigvals(branching_matrix))))

        ll = -result.fun
        elapsed = (time.perf_counter() - t0) * 1000.0

        hawkes_result = HawkesResult(
            mu=mu,
            alpha=alpha,
            beta=beta,
            branching_ratio=branching_ratio,
            log_likelihood=ll,
            n_events=[len(ev) for ev in events],
        )
        return SPTResult(
            data=hawkes_result,
            metadata={
                "method": "MLE_exponential_kernel",
                "converged": result.success,
                "optimizer_message": result.message,
                "n_iterations": result.nit,
            },
            computation_time_ms=elapsed,
        )

    def intensity(
        self,
        t: float,
        events: list[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        r"""Compute the intensity vector λ(t) given past events.

        Parameters
        ----------
        t : float
            Time at which to evaluate the intensity.
        events : list of ndarray
            Past event times per dimension.

        Returns
        -------
        ndarray of shape (d,)
            Intensity λ_i(t) for each dimension.
        """
        require(self._fitted, "Must call .fit() before .intensity()")
        assert (
            self._mu is not None and self._alpha is not None and self._beta is not None
        )
        d = self._d
        lam = self._mu.copy()
        for i in range(d):
            for j in range(d):
                past = events[j][events[j] < t]
                if len(past) > 0:
                    lam[i] += self._alpha[i, j] * np.sum(
                        self._beta[i, j] * np.exp(-self._beta[i, j] * (t - past))
                    )
        return lam

    # -- internal ---------------------------------------------------------------

    def _unpack(
        self, params: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        d = self._d
        mu = params[:d]
        alpha = params[d : d + d * d].reshape(d, d)
        beta = params[d + d * d :].reshape(d, d)
        return mu, alpha, beta

    def _log_likelihood(
        self,
        events: list[NDArray[np.float64]],
        T: float,
        mu: NDArray[np.float64],
        alpha: NDArray[np.float64],
        beta: NDArray[np.float64],
    ) -> float:
        r"""Compute log-likelihood of a Hawkes process.

        For each dimension i:
          LL_i = Σ_k log λ_i(t_{ik}) − ∫₀ᵀ λ_i(s) ds

        .. note::
           This implementation is O(N²) per dimension because it scans
           all past events at each event time.  An Ozaki (1979) recursive
           formulation would reduce this to O(N); see TODO below.

        TODO: Implement Ozaki recursive computation for O(N) performance.
        """
        d = self._d
        ll = 0.0

        for i in range(d):
            ev_i = events[i]

            for k in range(len(ev_i)):
                t_ik = ev_i[k]
                lam_at_t = mu[i]

                for j in range(d):
                    ev_j = events[j]
                    past = ev_j[ev_j < t_ik]
                    if len(past) > 0:
                        lam_at_t += alpha[i, j] * np.sum(
                            beta[i, j] * np.exp(-beta[i, j] * (t_ik - past))
                        )

                if lam_at_t > 0:
                    ll += np.log(lam_at_t)
                else:
                    ll += np.log(1e-300)

            ll -= mu[i] * T

            for j in range(d):
                ev_j = events[j]
                for t_jk in ev_j:
                    ll -= alpha[i, j] * (1.0 - np.exp(-beta[i, j] * (T - t_jk)))

        return ll


def simulate_hawkes(
    mu: NDArray[np.float64],
    alpha: NDArray[np.float64],
    beta: NDArray[np.float64],
    T: float,
    *,
    seed: int | None = None,
) -> SPTResult[list[NDArray[np.float64]]]:
    r"""Simulate a multivariate Hawkes process via Ogata's thinning algorithm.

    Algorithm (Ogata, 1981):
      1. Compute upper bound λ̄ on the total intensity.
      2. Sample candidate inter-arrival Δt ~ Exp(λ̄).
      3. Compute true intensity λ(t_candidate).
      4. Accept with probability λ(t_candidate) / λ̄; otherwise reject.
      5. Repeat until t > T.

    Parameters
    ----------
    mu : ndarray of shape (d,)
        Baseline intensities.
    alpha : ndarray of shape (d, d)
        Excitation magnitudes.
    beta : ndarray of shape (d, d)
        Decay rates.
    T : float
        Simulation horizon [0, T].
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SPTResult[list[ndarray]]
        List of d sorted event-time arrays.

    References
    ----------
    Ogata (1981), "On Lewis' simulation method for point processes,"
    IEEE Trans. Inform. Theory 27(1), pp. 23-31.
    """
    t0_wall = time.perf_counter()
    mu = np.asarray(mu, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    d = len(mu)
    require(alpha.shape == (d, d), f"alpha shape must be ({d},{d})")
    require(beta.shape == (d, d), f"beta shape must be ({d},{d})")
    require(T > 0, "T must be positive")

    rng = np.random.default_rng(seed)
    events: list[list[float]] = [[] for _ in range(d)]
    t = 0.0
    lam_bar = float(mu.sum()) + float(alpha.sum())
    max_events = int(lam_bar * T * 10) + 10000

    n_generated = 0
    while t < T and n_generated < max_events:
        dt = rng.exponential(1.0 / max(lam_bar, 1e-10))
        t += dt
        if t >= T:
            break

        lam = mu.copy()
        for i in range(d):
            for j in range(d):
                for t_jk in events[j]:
                    lam[i] += (
                        alpha[i, j] * beta[i, j] * np.exp(-beta[i, j] * (t - t_jk))
                    )

        total_lam = float(lam.sum())
        lam_bar = max(total_lam, float(mu.sum()))

        u = rng.uniform()
        if u * lam_bar <= total_lam and total_lam > 0:
            probs = lam / total_lam
            dim = rng.choice(d, p=probs)
            events[dim].append(t)
            lam_bar = total_lam + float((alpha[:, dim] * beta[:, dim]).sum())
            n_generated += 1

    result = [np.array(ev, dtype=np.float64) for ev in events]
    elapsed = (time.perf_counter() - t0_wall) * 1000.0
    return SPTResult(
        data=result,
        metadata={
            "method": "Ogata_thinning",
            "T": T,
            "n_events": [len(ev) for ev in result],
        },
        computation_time_ms=elapsed,
    )
