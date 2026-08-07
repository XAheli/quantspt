"""Tests for core extensions: AutoDiffGeneratingFunction, StochasticProcessArray, CovarianceRate."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.core.covariance import (
    ConstantCovarianceRate,
    CovarianceRateProcess,
    RollingCovarianceRate,
    relative_covariance,
)
from quantspt.core.generating_functions import (
    AutoDiffGeneratingFunction,
    DiversityGenerator,
    drift_process,
)
from quantspt.core.processes import (
    CorrelatedGBM,
    JointProcess,
    StochasticProcessArray,
    simulate_path,
)
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# AutoDiffGeneratingFunction tests
# ---------------------------------------------------------------------------


class TestAutoDiffGeneratingFunction:
    def test_evaluation(self) -> None:
        def G(mu):
            return float(np.sum(mu**0.5))

        gen = AutoDiffGeneratingFunction(G, name_str="sqrt_sum", backend="finite_diff")
        mu = np.array([0.3, 0.3, 0.4])
        assert gen(mu) == pytest.approx(np.sum(mu**0.5), rel=1e-10)

    def test_name_property(self) -> None:
        gen = AutoDiffGeneratingFunction(
            lambda mu: 1.0, name_str="const", backend="finite_diff"
        )
        assert gen.name == "const"

    def test_log_gradient_finite_diff(self) -> None:
        p = 0.5
        diversity = DiversityGenerator(p)

        def G(mu):
            return float(np.sum(mu**p)) ** (1.0 / p)

        gen = AutoDiffGeneratingFunction(G, backend="finite_diff", h=1e-7)
        mu = np.array([0.3, 0.3, 0.4])
        auto_grad = gen.log_gradient(mu)
        analytical_grad = diversity.log_gradient(mu)
        assert_allclose(auto_grad, analytical_grad, atol=1e-4)

    def test_hessian_finite_diff(self) -> None:
        p = 0.5

        def G(mu):
            return float(np.sum(mu**p)) ** (1.0 / p)

        diversity = DiversityGenerator(p)
        gen = AutoDiffGeneratingFunction(G, backend="finite_diff", h=1e-5)
        mu = np.array([0.3, 0.3, 0.4])
        auto_hess = gen.hessian(mu)
        analytical_hess = diversity.hessian(mu)
        assert_allclose(auto_hess, analytical_hess, atol=1e-2)

    def test_weights_consistent_with_analytical(self) -> None:
        p = 0.7
        diversity = DiversityGenerator(p)

        def G(mu):
            return float(np.sum(mu**p)) ** (1.0 / p)

        gen = AutoDiffGeneratingFunction(G, backend="finite_diff", h=1e-7)
        mu = np.array([0.2, 0.3, 0.5])
        auto_weights = gen.weights(mu)
        analytical_weights = diversity.weights(mu)
        assert_allclose(auto_weights, analytical_weights, atol=1e-4)

    def test_drift_consistent(self) -> None:
        p = 0.5
        diversity = DiversityGenerator(p)

        def G(mu):
            return float(np.sum(mu**p)) ** (1.0 / p)

        gen = AutoDiffGeneratingFunction(G, backend="finite_diff", h=1e-5)
        mu = np.array([0.3, 0.3, 0.4])
        a = np.diag([0.04, 0.09, 0.06])
        tau = relative_covariance(a, mu)
        auto_drift = drift_process(gen, mu, tau)
        analytical_drift = drift_process(diversity, mu, tau)
        assert_allclose(auto_drift, analytical_drift, atol=1e-2)

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="backend"):
            AutoDiffGeneratingFunction(lambda mu: 1.0, backend="cuda")

    def test_auto_backend_falls_back(self) -> None:
        gen = AutoDiffGeneratingFunction(lambda mu: float(np.sum(mu)), backend="auto")
        mu = np.array([0.5, 0.5])
        result = gen(mu)
        assert result == pytest.approx(1.0)

    def test_hessian_symmetric(self) -> None:
        def G(mu):
            return float(np.prod(mu**0.3))

        gen = AutoDiffGeneratingFunction(G, backend="finite_diff")
        mu = np.array([0.25, 0.25, 0.25, 0.25])
        H = gen.hessian(mu)
        assert_allclose(H, H.T, atol=1e-6)


# ---------------------------------------------------------------------------
# StochasticProcessArray tests
# ---------------------------------------------------------------------------


class TestStochasticProcessArray:
    def test_basic_construction(self) -> None:
        p1 = CorrelatedGBM(
            mu=np.array([0.05, 0.08]),
            cov=np.array([[0.04, 0.01], [0.01, 0.09]]),
            x0=np.array([100.0, 50.0]),
        )
        p2 = CorrelatedGBM(
            mu=np.array([0.03]),
            cov=np.array([[0.02]]),
            x0=np.array([200.0]),
        )
        arr = StochasticProcessArray(processes=[p1, p2])
        assert arr.size() == 3
        assert arr.factors() == 3

    def test_initial_values(self) -> None:
        p1 = CorrelatedGBM(
            mu=np.array([0.05]),
            cov=np.array([[0.04]]),
            x0=np.array([100.0]),
        )
        p2 = CorrelatedGBM(
            mu=np.array([0.08]),
            cov=np.array([[0.09]]),
            x0=np.array([50.0]),
        )
        arr = StochasticProcessArray(processes=[p1, p2])
        assert_allclose(arr.initial_values(), [100.0, 50.0])

    def test_evolve_produces_positive(self) -> None:
        rng = np.random.default_rng(42)
        p = CorrelatedGBM(
            mu=np.array([0.05, 0.08]),
            cov=np.array([[0.04, 0.01], [0.01, 0.09]]),
            x0=np.array([100.0, 50.0]),
        )
        arr = StochasticProcessArray(processes=[p])
        x0 = arr.initial_values()
        dt = 1.0 / 252.0
        dw = rng.standard_normal(arr.factors()) * np.sqrt(dt)
        x1 = arr.evolve(0.0, x0, dt, dw)
        assert np.all(x1 > 0)

    def test_pre_evolve_hook(self) -> None:
        p = CorrelatedGBM(
            mu=np.array([0.05]),
            cov=np.array([[0.04]]),
            x0=np.array([100.0]),
        )
        called = [False]

        def pre_hook(t, x, dt):
            called[0] = True
            return x

        arr = StochasticProcessArray(processes=[p], pre_evolve=pre_hook)
        x0 = arr.initial_values()
        dw = np.array([0.01])
        arr.evolve(0.0, x0, 0.01, dw)
        assert called[0]

    def test_post_evolve_hook(self) -> None:
        p = CorrelatedGBM(
            mu=np.array([0.05, 0.08]),
            cov=np.array([[0.04, 0.01], [0.01, 0.09]]),
            x0=np.array([100.0, 50.0]),
        )

        def clamp_positive(t, x, dt):
            return np.maximum(x, 0.01)

        arr = StochasticProcessArray(processes=[p], post_evolve=clamp_positive)
        x0 = arr.initial_values()
        dw = np.array([-10.0, -10.0])
        x1 = arr.evolve(0.0, x0, 1.0, dw)
        assert np.all(x1 >= 0.01)

    def test_simulate_path_integration(self) -> None:
        rng = np.random.default_rng(42)
        p = CorrelatedGBM(
            mu=np.array([0.05]),
            cov=np.array([[0.04]]),
            x0=np.array([100.0]),
        )
        arr = StochasticProcessArray(processes=[p])
        times, path = simulate_path(arr, T=1.0, n_steps=100, rng=rng)
        assert times.shape == (101,)
        assert path.shape == (101, 1)
        assert np.all(path > 0)

    def test_empty_processes_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="at least one"):
            StochasticProcessArray(processes=[])

    def test_drift_shape(self) -> None:
        p1 = CorrelatedGBM(
            mu=np.array([0.05, 0.08]),
            cov=np.array([[0.04, 0.01], [0.01, 0.09]]),
            x0=np.array([100.0, 50.0]),
        )
        arr = StochasticProcessArray(processes=[p1])
        x = arr.initial_values()
        drift = arr.drift(0.0, x)
        assert drift.shape == (2,)

    def test_diffusion_shape(self) -> None:
        p1 = CorrelatedGBM(
            mu=np.array([0.05, 0.08]),
            cov=np.array([[0.04, 0.01], [0.01, 0.09]]),
            x0=np.array([100.0, 50.0]),
        )
        arr = StochasticProcessArray(processes=[p1])
        x = arr.initial_values()
        sigma = arr.diffusion(0.0, x)
        assert sigma.shape == (2, 2)


# ---------------------------------------------------------------------------
# JointProcess tests
# ---------------------------------------------------------------------------


class TestJointProcess:
    def test_basic_joint_process(self) -> None:
        def drift(t, x):
            return 0.05 * x

        def diffusion(t, x):
            return np.diag(0.2 * x)

        jp = JointProcess(
            drift_fn=drift,
            diffusion_fn=diffusion,
            x0=np.array([100.0, 50.0]),
            n_factors=2,
        )
        assert jp.size() == 2
        assert jp.factors() == 2

    def test_evolve_euler(self) -> None:
        def drift(t, x):
            return np.zeros_like(x)

        def diffusion(t, x):
            return np.eye(2) * 0.1

        jp = JointProcess(
            drift_fn=drift,
            diffusion_fn=diffusion,
            x0=np.array([1.0, 1.0]),
            n_factors=2,
        )
        dw = np.array([0.01, -0.01])
        x1 = jp.evolve(0.0, jp.initial_values(), 0.01, dw)
        assert x1.shape == (2,)

    def test_pre_post_hooks(self) -> None:
        def drift(t, x):
            return np.array([0.0, 0.0])

        def diffusion(t, x):
            return np.eye(2)

        pre_called = [False]
        post_called = [False]

        def pre(t, x, dt):
            pre_called[0] = True
            return x

        def post(t, x, dt):
            post_called[0] = True
            return np.abs(x)

        jp = JointProcess(
            drift_fn=drift,
            diffusion_fn=diffusion,
            x0=np.array([1.0, 1.0]),
            n_factors=2,
            pre_evolve=pre,
            post_evolve=post,
        )
        dw = np.array([0.1, 0.1])
        jp.evolve(0.0, jp.initial_values(), 0.01, dw)
        assert pre_called[0]
        assert post_called[0]

    def test_simulate_path_integration(self) -> None:
        rng = np.random.default_rng(42)

        def drift(t, x):
            return 0.05 * x

        def diffusion(t, x):
            return np.diag(0.2 * x)

        jp = JointProcess(
            drift_fn=drift,
            diffusion_fn=diffusion,
            x0=np.array([100.0]),
            n_factors=1,
        )
        times, path = simulate_path(jp, T=1.0, n_steps=100, rng=rng)
        assert times.shape == (101,)
        assert path.shape == (101, 1)

    def test_invalid_x0_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="1-D"):
            JointProcess(
                drift_fn=lambda t, x: x,
                diffusion_fn=lambda t, x: np.eye(2),
                x0=np.array([[1.0, 2.0]]),
                n_factors=2,
            )


# ---------------------------------------------------------------------------
# CovarianceRateProcess tests
# ---------------------------------------------------------------------------


class TestCovarianceRateProcess:
    def test_constant_covariance(self) -> None:
        a = np.array([[0.04, 0.01], [0.01, 0.09]])
        process = ConstantCovarianceRate(a)
        result = process.covariance_at(0.0)
        assert_allclose(result, a)

    def test_constant_time_invariant(self) -> None:
        a = np.diag([0.04, 0.09, 0.06])
        process = ConstantCovarianceRate(a)
        assert_allclose(process.covariance_at(0.0), process.covariance_at(100.0))

    def test_constant_n_assets(self) -> None:
        a = np.eye(5) * 0.04
        process = ConstantCovarianceRate(a)
        assert process.n_assets() == 5

    def test_constant_returns_copy(self) -> None:
        a = np.diag([0.04, 0.09])
        process = ConstantCovarianceRate(a)
        result = process.covariance_at(0.0)
        result[0, 0] = 999.0
        assert process.covariance_at(0.0)[0, 0] == 0.04

    def test_protocol_compliance(self) -> None:
        a = np.eye(3) * 0.04
        process = ConstantCovarianceRate(a)
        assert isinstance(process, CovarianceRateProcess)

    def test_rolling_nearest(self) -> None:
        times = np.array([0.0, 1.0, 2.0])
        covs = np.array(
            [
                np.eye(2) * 0.04,
                np.eye(2) * 0.09,
                np.eye(2) * 0.16,
            ]
        )
        process = RollingCovarianceRate(times, covs, interpolation="nearest")
        result = process.covariance_at(0.8)
        assert_allclose(result, np.eye(2) * 0.09)

    def test_rolling_linear(self) -> None:
        times = np.array([0.0, 1.0])
        covs = np.array(
            [
                np.eye(2) * 0.04,
                np.eye(2) * 0.16,
            ]
        )
        process = RollingCovarianceRate(times, covs, interpolation="linear")
        result = process.covariance_at(0.5)
        expected = np.eye(2) * 0.10
        assert_allclose(result, expected, atol=1e-10)

    def test_rolling_extrapolation(self) -> None:
        times = np.array([1.0, 2.0])
        covs = np.array([np.eye(2) * 0.04, np.eye(2) * 0.09])
        process = RollingCovarianceRate(times, covs, interpolation="linear")
        result_before = process.covariance_at(0.0)
        assert_allclose(result_before, np.eye(2) * 0.04)
        result_after = process.covariance_at(5.0)
        assert_allclose(result_after, np.eye(2) * 0.09)

    def test_rolling_n_assets(self) -> None:
        times = np.array([0.0])
        covs = np.array([np.eye(4) * 0.05])
        process = RollingCovarianceRate(times, covs)
        assert process.n_assets() == 4

    def test_rolling_protocol_compliance(self) -> None:
        times = np.array([0.0])
        covs = np.array([np.eye(2) * 0.04])
        process = RollingCovarianceRate(times, covs)
        assert isinstance(process, CovarianceRateProcess)

    def test_invalid_interpolation_raises(self) -> None:
        times = np.array([0.0])
        covs = np.array([np.eye(2) * 0.04])
        with pytest.raises(SPTInvariantError, match="interpolation"):
            RollingCovarianceRate(times, covs, interpolation="cubic")

    def test_duplicate_times_linear_no_crash(self) -> None:
        """Duplicate timestamps must not cause division by zero in linear interp."""
        times = np.array([0.0, 0.5, 0.5, 1.0])
        covs = np.array(
            [
                np.eye(2) * 0.04,
                np.eye(2) * 0.09,
                np.eye(2) * 0.09,
                np.eye(2) * 0.16,
            ]
        )
        process = RollingCovarianceRate(times, covs, interpolation="linear")
        result = process.covariance_at(0.5)
        assert np.all(np.isfinite(result))
        assert_allclose(result, np.eye(2) * 0.09)

    def test_all_duplicate_times_linear(self) -> None:
        """All-identical timestamps should return a valid covariance."""
        times = np.array([1.0, 1.0])
        covs = np.array([np.eye(2) * 0.04, np.eye(2) * 0.09])
        process = RollingCovarianceRate(times, covs, interpolation="linear")
        result = process.covariance_at(1.0)
        assert np.all(np.isfinite(result))
