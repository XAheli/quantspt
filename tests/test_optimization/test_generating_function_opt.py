"""Tests for optimization/generating_function -- parameter optimisation.

Validates that the generating function parameter optimiser correctly
identifies optimal parameters, handles grid search and refinement,
and supports both objective functions.

Mathematical References
-----------------------
- FGP drift process: F&K Survey Eq. 11.3
- Diversity generator G_p: F&K Survey Remark 11.1
- Master formula: F&K Survey Eq. 11.2
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.core.generating_functions import (
    DiversityGenerator,
    GeneratingFunction,
)
from quantspt.errors import SPTInvariantError
from quantspt.optimization.generating_function import (
    OptimizationResult,
    optimize_diversity_parameter,
    optimize_generator_parameter,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def diverse_market_data(
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Synthetic diverse-market data: weights + covariance matrices."""
    n = 5
    T = 100
    weights = np.zeros((T, n))
    cov_matrices = []

    base_weights = rng.dirichlet(np.ones(n) * 5)
    for t in range(T):
        noise = rng.standard_normal(n) * 0.01
        w = np.maximum(base_weights + noise, 0.01)
        w /= w.sum()
        weights[t] = w
        cov = 0.04 * np.eye(n) + 0.005 * rng.standard_normal((n, n))
        cov = (cov + cov.T) / 2
        cov += np.eye(n) * 0.05
        cov_matrices.append(cov)

    return weights, cov_matrices


# =========================================================================
# A. optimize_diversity_parameter
# =========================================================================


class TestOptimizeDiversityParameter:
    """Tests for optimize_diversity_parameter()."""

    def test_returns_optimization_result(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        weights, covs = diverse_market_data
        result = optimize_diversity_parameter(weights, covs, n_grid=20, refine=False)
        assert isinstance(result, OptimizationResult)

    def test_optimal_p_in_range(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        """Optimal p should be within the search range."""
        weights, covs = diverse_market_data
        result = optimize_diversity_parameter(
            weights, covs, p_range=(0.2, 0.8), n_grid=20, refine=False
        )
        assert 0.2 <= result.optimal_param <= 0.8

    def test_optimal_value_positive(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        """For diverse markets, mean drift should be positive."""
        weights, covs = diverse_market_data
        result = optimize_diversity_parameter(weights, covs, n_grid=20, refine=False)
        assert result.optimal_value > 0

    def test_grid_data_stored(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        weights, covs = diverse_market_data
        result = optimize_diversity_parameter(weights, covs, n_grid=30, refine=False)
        assert len(result.grid_params) == 30
        assert len(result.grid_values) == 30

    def test_refine_improves_or_matches(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        """Refinement should improve or match the grid-only optimum."""
        weights, covs = diverse_market_data
        grid_only = optimize_diversity_parameter(weights, covs, n_grid=20, refine=False)
        refined = optimize_diversity_parameter(weights, covs, n_grid=20, refine=True)
        assert refined.optimal_value >= grid_only.optimal_value - 1e-10

    def test_method_field(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        weights, covs = diverse_market_data
        grid_only = optimize_diversity_parameter(weights, covs, n_grid=20, refine=False)
        assert grid_only.method == "grid"

    def test_sharpe_objective(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        """Sharpe objective should also work."""
        weights, covs = diverse_market_data
        result = optimize_diversity_parameter(
            weights, covs, n_grid=15, objective="sharpe", refine=False
        )
        assert isinstance(result, OptimizationResult)
        assert 0.1 <= result.optimal_param <= 0.9

    def test_invalid_p_range(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        weights, covs = diverse_market_data
        with pytest.raises(SPTInvariantError):
            optimize_diversity_parameter(weights, covs, p_range=(0.0, 0.5))
        with pytest.raises(SPTInvariantError):
            optimize_diversity_parameter(weights, covs, p_range=(0.5, 1.0))

    def test_grid_values_correspond_to_params(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        """Grid params should be evenly spaced in the range."""
        weights, covs = diverse_market_data
        result = optimize_diversity_parameter(
            weights, covs, p_range=(0.2, 0.8), n_grid=10, refine=False
        )
        expected = np.linspace(0.2, 0.8, 10)
        assert_allclose(result.grid_params, expected, atol=1e-14)


# =========================================================================
# B. optimize_generator_parameter (generic)
# =========================================================================


class TestOptimizeGeneratorParameter:
    """Tests for the generic optimize_generator_parameter()."""

    def test_custom_factory(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        """Should work with any GeneratingFunction factory."""
        weights, covs = diverse_market_data

        def factory(p: float) -> GeneratingFunction:
            return DiversityGenerator(p)

        result = optimize_generator_parameter(
            factory,
            weights,
            covs,
            param_range=(0.1, 0.9),
            n_grid=10,
            refine=False,
        )
        assert isinstance(result, OptimizationResult)

    def test_higher_grid_resolution_better_result(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        """Finer grid should give >= objective value."""
        weights, covs = diverse_market_data

        def factory(p: float) -> GeneratingFunction:
            return DiversityGenerator(p)

        coarse = optimize_generator_parameter(
            factory, weights, covs, n_grid=5, refine=False
        )
        fine = optimize_generator_parameter(
            factory, weights, covs, n_grid=50, refine=False
        )
        assert fine.optimal_value >= coarse.optimal_value - 1e-8

    def test_invalid_param_range(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        weights, covs = diverse_market_data

        def factory(p: float) -> GeneratingFunction:
            return DiversityGenerator(p)

        with pytest.raises(SPTInvariantError, match="param_range"):
            optimize_generator_parameter(factory, weights, covs, param_range=(0.9, 0.1))

    def test_invalid_objective(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        weights, covs = diverse_market_data

        def factory(p: float) -> GeneratingFunction:
            return DiversityGenerator(p)

        with pytest.raises(SPTInvariantError, match="Unknown objective"):
            optimize_generator_parameter(
                factory,
                weights,
                covs,
                objective="invalid",  # type: ignore[arg-type]
            )

    def test_1d_weights_rejected(self) -> None:
        def factory(p: float) -> GeneratingFunction:
            return DiversityGenerator(p)

        with pytest.raises(SPTInvariantError, match="2-D"):
            optimize_generator_parameter(factory, np.array([0.5, 0.5]), [np.eye(2)])

    def test_handles_factory_errors_gracefully(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        """If factory raises for some params, those should get -inf."""
        weights, covs = diverse_market_data

        def bad_factory(p: float) -> GeneratingFunction:
            if p < 0.05:
                raise ValueError("Too small")
            return DiversityGenerator(min(max(p, 0.01), 0.99))

        result = optimize_generator_parameter(
            bad_factory,
            weights,
            covs,
            param_range=(0.01, 0.5),
            n_grid=10,
            refine=False,
        )
        assert result.optimal_param >= 0.05


# =========================================================================
# C. OptimizationResult dataclass
# =========================================================================


class TestOptimizationResult:
    """Tests for the OptimizationResult dataclass."""

    def test_frozen(self) -> None:
        result = OptimizationResult(
            optimal_param=0.5,
            optimal_value=0.01,
            grid_params=np.linspace(0.1, 0.9, 10),
            grid_values=np.ones(10) * 0.01,
            method="grid",
        )
        with pytest.raises(AttributeError):
            result.optimal_param = 0.6  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        result = OptimizationResult(
            optimal_param=0.42,
            optimal_value=0.015,
            grid_params=np.array([0.1, 0.5, 0.9]),
            grid_values=np.array([0.01, 0.015, 0.005]),
            method="grid+brent",
        )
        assert result.optimal_param == 0.42
        assert result.optimal_value == 0.015
        assert result.method == "grid+brent"
        assert len(result.grid_params) == 3


# =========================================================================
# D. Integration: end-to-end parameter selection
# =========================================================================


class TestIntegration:
    """End-to-end test: optimise p, then use it to generate weights."""

    def test_optimal_p_produces_valid_weights(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        """Weights from optimal p should sum to 1 and be non-negative."""
        weights, covs = diverse_market_data
        result = optimize_diversity_parameter(weights, covs, n_grid=15, refine=True)

        G = DiversityGenerator(result.optimal_param)
        mu = weights[-1]
        pi = G.weights(mu)

        assert_allclose(np.sum(pi), 1.0, atol=1e-12)
        assert np.all(pi >= -1e-12)

    def test_optimal_drift_higher_than_endpoints(
        self, diverse_market_data: tuple[np.ndarray, list[np.ndarray]]
    ) -> None:
        """Optimal p should give drift >= boundary values."""
        weights, covs = diverse_market_data
        result = optimize_diversity_parameter(
            weights, covs, p_range=(0.2, 0.8), n_grid=20, refine=True
        )

        assert result.optimal_value >= result.grid_values[0] - 1e-10
        assert result.optimal_value >= result.grid_values[-1] - 1e-10


# =========================================================================
# E. Coverage: ndarray cov, zero-std, refinement
# =========================================================================


class TestNdarrayCovMatrices:
    """Test with 3-D ndarray covariance matrices (vs list)."""

    def test_mean_drift_with_ndarray_cov(self) -> None:
        """Passing cov_matrices as a 3-D ndarray should work."""
        rng = np.random.default_rng(42)
        n = 3
        T = 20
        weights = rng.dirichlet(np.ones(n), size=T)
        cov_3d = np.array([np.diag(rng.uniform(0.01, 0.1, n)) for _ in range(T)])

        result = optimize_diversity_parameter(weights, cov_3d, n_grid=5, refine=False)
        assert result.optimal_param > 0

    def test_sharpe_with_ndarray_cov(self) -> None:
        """Sharpe objective with 3-D ndarray cov_matrices."""
        rng = np.random.default_rng(42)
        n = 3
        T = 20
        weights = rng.dirichlet(np.ones(n), size=T)
        cov_3d = np.array([np.diag(rng.uniform(0.01, 0.1, n)) for _ in range(T)])

        result = optimize_diversity_parameter(
            weights, cov_3d, n_grid=5, refine=False, objective="sharpe"
        )
        assert result.optimal_param > 0


class TestZeroStdFallback:
    """Cover the zero-std branch in _sharpe_of_relative_return."""

    def test_constant_drift_sharpe(self) -> None:
        """When all drifts are identical, std=0 and the fallback triggers."""
        T = 10
        mu = np.array([0.5, 0.3, 0.2])
        weights = np.tile(mu, (T, 1))
        cov = np.diag([0.04, 0.04, 0.04])
        covs = [cov] * T

        result = optimize_generator_parameter(
            lambda p: DiversityGenerator(p),
            weights,
            covs,
            param_range=(0.3, 0.7),
            n_grid=3,
            refine=False,
            objective="sharpe",
        )
        assert result.optimal_value >= 0


class TestRefinementPath:
    """Ensure the Brent refinement path is exercised."""

    def test_refinement_improves_or_matches(self) -> None:
        """With refine=True, the result should be >= grid-only result."""
        rng = np.random.default_rng(42)
        n = 4
        T = 30
        weights = rng.dirichlet(np.ones(n), size=T)
        covs = [np.diag(rng.uniform(0.02, 0.08, n)) for _ in range(T)]

        result_no_refine = optimize_diversity_parameter(
            weights, covs, n_grid=5, refine=False
        )
        result_refine = optimize_diversity_parameter(
            weights, covs, n_grid=5, refine=True
        )
        assert result_refine.optimal_value >= result_no_refine.optimal_value - 1e-10
        assert result_refine.method in ("grid", "grid+brent")
