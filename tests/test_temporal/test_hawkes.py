"""Tests for quantspt.temporal.hawkes — Hawkes process estimation and simulation.

Validates parameter recovery, branching ratio computation, and the
Ogata thinning simulation algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt._result import SPTResult
from quantspt.errors import SPTInvariantError
from quantspt.temporal.hawkes import (
    HawkesProcess,
    HawkesResult,
    simulate_hawkes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def known_hawkes_params() -> dict:
    """Known 1-D Hawkes parameters for recovery tests."""
    return {
        "mu": np.array([0.5]),
        "alpha": np.array([[0.3]]),
        "beta": np.array([[1.5]]),
    }


@pytest.fixture()
def simulated_events(known_hawkes_params: dict) -> tuple[list[np.ndarray], float]:
    """Simulate events from known params for parameter recovery."""
    p = known_hawkes_params
    T = 2000.0
    result = simulate_hawkes(p["mu"], p["alpha"], p["beta"], T, seed=42)
    return result.data, T


@pytest.fixture()
def bivariate_params() -> dict:
    """Known 2-D Hawkes parameters."""
    return {
        "mu": np.array([0.5, 0.3]),
        "alpha": np.array([[0.3, 0.1], [0.05, 0.2]]),
        "beta": np.array([[1.5, 1.0], [1.0, 1.5]]),
    }


# ---------------------------------------------------------------------------
# Simulation Tests
# ---------------------------------------------------------------------------


class TestHawkesSimulation:
    def test_result_type(self, known_hawkes_params: dict) -> None:
        p = known_hawkes_params
        result = simulate_hawkes(p["mu"], p["alpha"], p["beta"], T=10.0, seed=1)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, list)
        assert len(result.data) == 1

    def test_events_sorted(self, known_hawkes_params: dict) -> None:
        p = known_hawkes_params
        result = simulate_hawkes(p["mu"], p["alpha"], p["beta"], T=100.0, seed=42)
        for ev in result.data:
            if len(ev) > 1:
                assert np.all(np.diff(ev) >= 0), "Events must be sorted"

    def test_events_in_window(self, known_hawkes_params: dict) -> None:
        p = known_hawkes_params
        T = 50.0
        result = simulate_hawkes(p["mu"], p["alpha"], p["beta"], T=T, seed=42)
        for ev in result.data:
            if len(ev) > 0:
                assert ev[0] >= 0
                assert ev[-1] <= T

    def test_generates_events(self, known_hawkes_params: dict) -> None:
        p = known_hawkes_params
        result = simulate_hawkes(p["mu"], p["alpha"], p["beta"], T=100.0, seed=42)
        total = sum(len(ev) for ev in result.data)
        assert total > 10, "Should generate a reasonable number of events"

    def test_reproducibility(self, known_hawkes_params: dict) -> None:
        p = known_hawkes_params
        r1 = simulate_hawkes(p["mu"], p["alpha"], p["beta"], T=50.0, seed=123)
        r2 = simulate_hawkes(p["mu"], p["alpha"], p["beta"], T=50.0, seed=123)
        for ev1, ev2 in zip(r1.data, r2.data, strict=True):
            assert_allclose(ev1, ev2)

    def test_bivariate_simulation(self, bivariate_params: dict) -> None:
        p = bivariate_params
        result = simulate_hawkes(p["mu"], p["alpha"], p["beta"], T=100.0, seed=42)
        assert len(result.data) == 2
        assert all(isinstance(ev, np.ndarray) for ev in result.data)

    def test_higher_mu_more_events(self) -> None:
        """Higher baseline intensity → more events."""
        low = simulate_hawkes(
            np.array([0.1]), np.array([[0.1]]), np.array([[1.0]]), T=100.0, seed=42
        )
        high = simulate_hawkes(
            np.array([5.0]), np.array([[0.1]]), np.array([[1.0]]), T=100.0, seed=42
        )
        assert sum(len(ev) for ev in high.data) > sum(len(ev) for ev in low.data)


# ---------------------------------------------------------------------------
# Parameter Recovery Tests
# ---------------------------------------------------------------------------


class TestHawkesEstimation:
    def test_parameter_recovery_univariate(
        self, known_hawkes_params: dict, simulated_events: tuple
    ) -> None:
        """Fit on simulated data should recover true parameters approximately."""
        events, T = simulated_events
        true = known_hawkes_params

        hp = HawkesProcess(n_dim=1)
        result = hp.fit(events, T)
        est = result.data

        assert isinstance(est, HawkesResult)
        assert est.mu[0] > 0, "Baseline intensity should be positive"
        assert est.alpha[0, 0] > 0, "Excitation should be positive"
        assert est.beta[0, 0] > 0, "Decay should be positive"
        assert_allclose(est.mu, true["mu"], rtol=1.0)

    def test_branching_ratio(
        self, known_hawkes_params: dict, simulated_events: tuple
    ) -> None:
        """Branching ratio should be < 1 and positive for self-exciting process."""
        events, T = simulated_events
        hp = HawkesProcess(n_dim=1)
        result = hp.fit(events, T)
        assert result.data.branching_ratio < 1.0, "Stationary => BR < 1"
        assert result.data.branching_ratio > 0.05, (
            "Self-exciting => BR meaningfully > 0"
        )

    def test_branching_ratio_below_one(
        self, known_hawkes_params: dict, simulated_events: tuple
    ) -> None:
        events, T = simulated_events
        hp = HawkesProcess(n_dim=1)
        result = hp.fit(events, T)
        assert result.data.branching_ratio < 1.0, "Stationary process has BR < 1"

    def test_log_likelihood_finite(
        self, known_hawkes_params: dict, simulated_events: tuple
    ) -> None:
        events, T = simulated_events
        hp = HawkesProcess(n_dim=1)
        result = hp.fit(events, T)
        assert np.isfinite(result.data.log_likelihood)

    def test_n_events_recorded(
        self, known_hawkes_params: dict, simulated_events: tuple
    ) -> None:
        events, T = simulated_events
        hp = HawkesProcess(n_dim=1)
        result = hp.fit(events, T)
        assert result.data.n_events == [len(events[0])]

    def test_result_type(
        self, known_hawkes_params: dict, simulated_events: tuple
    ) -> None:
        events, T = simulated_events
        hp = HawkesProcess(n_dim=1)
        result = hp.fit(events, T)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, HawkesResult)

    def test_intensity_after_fit(
        self, known_hawkes_params: dict, simulated_events: tuple
    ) -> None:
        events, T = simulated_events
        hp = HawkesProcess(n_dim=1)
        hp.fit(events, T)
        lam = hp.intensity(T / 2, events)
        assert lam.shape == (1,)
        assert lam[0] > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestHawkesValidation:
    def test_wrong_n_dim(self) -> None:
        hp = HawkesProcess(n_dim=2)
        with pytest.raises(SPTInvariantError, match="Expected 2"):
            hp.fit([np.array([1.0])], T=10.0)

    def test_negative_T(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            simulate_hawkes(
                np.array([1.0]), np.array([[0.5]]), np.array([[1.0]]), T=-1.0
            )

    def test_intensity_before_fit(self) -> None:
        hp = HawkesProcess(n_dim=1)
        with pytest.raises(SPTInvariantError, match="fit"):
            hp.intensity(1.0, [np.array([0.5])])


# ---------------------------------------------------------------------------
# Regression tests for critical bug fixes
# ---------------------------------------------------------------------------


class TestBranchingRatioFormula:
    """Branching ratio must equal spectral radius of alpha, NOT alpha/beta.

    The kernel is alpha * beta * exp(-beta * t), which integrates to
    alpha over [0, inf).  The old code computed alpha / beta, which is
    wrong: it would say a kernel with (alpha=0.3, beta=1.5) and one
    with (alpha=0.3, beta=0.1) have different branching ratios, but
    they both integrate to 0.3.
    """

    def test_branching_ratio_equals_alpha(self) -> None:
        """For 1-D process with alpha=0.3, branching ratio must be 0.3."""
        hp = HawkesProcess(n_dim=1)
        events, T = (
            simulate_hawkes(
                np.array([0.5]), np.array([[0.3]]), np.array([[1.5]]), T=2000.0, seed=42
            ).data,
            2000.0,
        )
        result = hp.fit(events, T)
        estimated_alpha = result.data.alpha[0, 0]
        assert_allclose(
            result.data.branching_ratio,
            estimated_alpha,
            rtol=1e-10,
        )

    def test_different_beta_same_branching_ratio(self) -> None:
        """Changing beta alone must NOT change the branching ratio.

        The old formula alpha/beta would give different values for
        different betas with the same alpha.  The correct formula
        (spectral radius of alpha) is independent of beta.
        """
        hp = HawkesProcess(n_dim=1)
        alpha = np.array([[0.3]])

        events1, T1 = (
            simulate_hawkes(
                np.array([0.5]), alpha, np.array([[0.5]]), T=2000.0, seed=1
            ).data,
            2000.0,
        )
        r1 = hp.fit(events1, T1)

        hp2 = HawkesProcess(n_dim=1)
        events2, T2 = (
            simulate_hawkes(
                np.array([0.5]), alpha, np.array([[5.0]]), T=2000.0, seed=2
            ).data,
            2000.0,
        )
        r2 = hp2.fit(events2, T2)

        assert abs(r1.data.alpha[0, 0] - r1.data.branching_ratio) < 1e-10
        assert abs(r2.data.alpha[0, 0] - r2.data.branching_ratio) < 1e-10


class TestOgataThinningBound:
    """Upper bound after accepting event in dim must be sum-of-products
    of alpha and beta for that dimension, NOT product-of-sums.

    Product-of-sums overestimates massively when alpha/beta have
    different magnitudes, causing the thinning to reject almost all
    candidates and waste computation.
    """

    def test_simulation_produces_reasonable_event_count(self) -> None:
        """With correct thinning bound, event count should match theory.

        For a 1-D stationary Hawkes with mu=1, alpha=0.3, beta=1.5,
        the stationary rate is mu / (1 - alpha) = 1 / 0.7 ~ 1.43.
        Over T=100, expect roughly 143 events, definitely > 50.
        """
        mu = np.array([1.0])
        alpha = np.array([[0.3]])
        beta = np.array([[1.5]])
        result = simulate_hawkes(mu, alpha, beta, T=100.0, seed=42)
        n_events = len(result.data[0])
        expected_rate = 1.0 / (1.0 - 0.3)
        assert n_events > expected_rate * 100 * 0.3, (
            f"Expected roughly {expected_rate * 100:.0f} events, got {n_events}"
        )
