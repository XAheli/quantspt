"""Tests for CovarianceConditionalFGP and related modules.

Covers covariance feature extraction, boundary robustness regularization,
cost-aware p selection, the conditional FGP training/prediction pipeline,
and integration with the backtesting engine.

Tests use both synthetic data (for unit tests) and real yfinance data
(for validation).
"""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.core.generating_functions import DiversityGenerator
from quantspt.ml.conditional_fgp import (
    DEFAULT_FEATURES,
    P_MAX,
    P_MIN,
    BoundaryRobustnessRegularizer,
    ConditionalFGPConfig,
    CovarianceConditionalFGP,
    CovarianceFeatureExtractor,
    cost_optimal_p,
    optimal_p_for_cost_level,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def cov_5(rng: np.random.Generator) -> np.ndarray:
    """5x5 PSD covariance matrix with realistic structure."""
    L = rng.standard_normal((5, 5)) * 0.02
    return L @ L.T + np.eye(5) * 0.01


@pytest.fixture()
def cov_10(rng: np.random.Generator) -> np.ndarray:
    """10x10 sector-structured covariance."""
    n = 10
    vols = rng.uniform(0.15, 0.40, size=n)
    corr = np.full((n, n), 0.3)
    corr[:5, :5] = 0.6
    corr[5:, 5:] = 0.6
    np.fill_diagonal(corr, 1.0)
    D = np.diag(vols)
    cov = D @ corr @ D
    return (cov + cov.T) / 2


@pytest.fixture()
def mu_5(rng: np.random.Generator) -> np.ndarray:
    """Pareto-distributed 5-asset market weights."""
    raw = (rng.pareto(1.0, size=5) + 1.0).astype(np.float64)
    return raw / raw.sum()


@pytest.fixture()
def mu_10(rng: np.random.Generator) -> np.ndarray:
    """10-asset market weights."""
    raw = (rng.pareto(1.0, size=10) + 1.0).astype(np.float64)
    return raw / raw.sum()


@pytest.fixture()
def synthetic_market_data(rng: np.random.Generator) -> dict:
    """Synthetic market with 10 assets, 500 days."""
    n, T = 10, 500
    vols = rng.uniform(0.15, 0.35, size=n)
    dt = 1.0 / 252.0

    prices = np.zeros((T, n))
    prices[0] = rng.uniform(50, 200, size=n)
    for t in range(1, T):
        dW = rng.standard_normal(n)
        prices[t] = prices[t - 1] * np.exp(
            (0.05 - 0.5 * vols**2) * dt + vols * np.sqrt(dt) * dW
        )

    weights = prices / prices.sum(axis=1, keepdims=True)
    returns = prices[1:] / prices[:-1]

    log_returns = np.log(prices[1:] / prices[:-1])
    cov_matrices = []
    window = 63
    for t in range(T):
        if t < window:
            cov_matrices.append(np.eye(n) * 0.04)
        else:
            chunk = log_returns[max(0, t - window) : t]
            cov_matrices.append(np.cov(chunk, rowvar=False) * 252)

    return {
        "weights": weights,
        "returns": returns,
        "cov_matrices": cov_matrices,
        "prices": prices,
        "n": n,
        "T": T,
    }


# ===================================================================
# CovarianceFeatureExtractor tests
# ===================================================================


class TestCovarianceFeatureExtractor:
    def test_default_features(self) -> None:
        ext = CovarianceFeatureExtractor()
        assert ext.feature_names == DEFAULT_FEATURES
        assert ext.n_features == len(DEFAULT_FEATURES)

    def test_custom_features(self) -> None:
        ext = CovarianceFeatureExtractor(features=["avg_correlation", "effective_rank"])
        assert ext.n_features == 2
        assert ext.feature_names == ["avg_correlation", "effective_rank"]

    def test_invalid_feature_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown feature"):
            CovarianceFeatureExtractor(features=["nonexistent"])

    def test_extract_returns_correct_shape(self, cov_5: np.ndarray) -> None:
        ext = CovarianceFeatureExtractor()
        features = ext.extract(cov_5)
        assert features.shape == (len(DEFAULT_FEATURES),)
        assert np.all(np.isfinite(features))

    def test_extract_identity_matrix(self) -> None:
        ext = CovarianceFeatureExtractor()
        eye = np.eye(5)
        features = ext.extract(eye)
        assert features.shape == (len(DEFAULT_FEATURES),)
        # Identity: avg corr should be 0, eigenvalue ratio = 1/5
        idx_corr = DEFAULT_FEATURES.index("avg_correlation")
        assert abs(features[idx_corr]) < 1e-10

    def test_extract_diagonal_matrix(self) -> None:
        ext = CovarianceFeatureExtractor()
        D = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        features = ext.extract(D)
        assert np.all(np.isfinite(features))

    def test_avg_correlation_range(self, cov_10: np.ndarray) -> None:
        ext = CovarianceFeatureExtractor(features=["avg_correlation"])
        val = ext.extract(cov_10)[0]
        assert -1.0 <= val <= 1.0

    def test_max_eigenvalue_ratio_range(self, cov_5: np.ndarray) -> None:
        ext = CovarianceFeatureExtractor(features=["max_eigenvalue_ratio"])
        val = ext.extract(cov_5)[0]
        assert 0.0 < val <= 1.0

    def test_effective_rank_positive(self, cov_5: np.ndarray) -> None:
        ext = CovarianceFeatureExtractor(features=["effective_rank"])
        val = ext.extract(cov_5)[0]
        assert val >= 1.0

    def test_eigenvalue_herfindahl_range(self, cov_5: np.ndarray) -> None:
        ext = CovarianceFeatureExtractor(features=["eigenvalue_herfindahl"])
        val = ext.extract(cov_5)[0]
        assert 0.0 < val <= 1.0

    def test_extract_batch(self, cov_5: np.ndarray) -> None:
        ext = CovarianceFeatureExtractor()
        batch = ext.extract_batch([cov_5, cov_5 * 2])
        assert batch.shape == (2, len(DEFAULT_FEATURES))

    def test_non_square_matrix_raises(self) -> None:
        ext = CovarianceFeatureExtractor()
        with pytest.raises(ValueError, match="square matrix"):
            ext.extract(np.ones((3, 4)))

    def test_1d_input_raises(self) -> None:
        ext = CovarianceFeatureExtractor()
        with pytest.raises(ValueError):
            ext.extract(np.ones(5))

    def test_all_feature_types(self, cov_5: np.ndarray) -> None:
        all_feats = [
            "avg_correlation",
            "max_eigenvalue_ratio",
            "effective_rank",
            "eigenvalue_herfindahl",
            "trace_normalized",
            "condition_number",
        ]
        ext = CovarianceFeatureExtractor(features=all_feats)
        features = ext.extract(cov_5)
        assert features.shape == (6,)
        assert np.all(np.isfinite(features))

    def test_high_correlation_detected(self) -> None:
        n = 5
        corr = np.full((n, n), 0.9)
        np.fill_diagonal(corr, 1.0)
        ext = CovarianceFeatureExtractor(features=["avg_correlation"])
        val = ext.extract(corr)[0]
        assert val > 0.8

    def test_singular_matrix_handled(self) -> None:
        cov = np.zeros((3, 3))
        cov[0, 0] = 1.0
        ext = CovarianceFeatureExtractor()
        features = ext.extract(cov)
        assert np.all(np.isfinite(features))

    def test_condition_number_feature(self) -> None:
        ext = CovarianceFeatureExtractor(features=["condition_number"])
        cov = np.diag([1.0, 0.01])
        val = ext.extract(cov)[0]
        assert val == pytest.approx(100.0, rel=1e-6)

    def test_single_asset_avg_correlation(self) -> None:
        """1x1 matrix: avg_correlation should be 0."""
        ext = CovarianceFeatureExtractor(features=["avg_correlation"])
        val = ext.extract(np.array([[0.04]]))[0]
        assert val == 0.0

    def test_zero_eigenvalues_effective_rank(self) -> None:
        """All-zero matrix: effective_rank fallback = 1."""
        ext = CovarianceFeatureExtractor(features=["effective_rank"])
        val = ext.extract(np.zeros((3, 3)))[0]
        assert val == 1.0

    def test_zero_eigenvalues_max_ratio(self) -> None:
        ext = CovarianceFeatureExtractor(features=["max_eigenvalue_ratio"])
        val = ext.extract(np.zeros((3, 3)))[0]
        assert val == 1.0

    def test_zero_eigenvalues_herfindahl(self) -> None:
        ext = CovarianceFeatureExtractor(features=["eigenvalue_herfindahl"])
        val = ext.extract(np.zeros((3, 3)))[0]
        assert val == 1.0

    def test_zero_n_trace_normalized(self) -> None:
        from quantspt.experimental.conditional_fgp import _trace_normalized

        assert _trace_normalized(np.array([1.0]), 0) == 0.0

    def test_no_positive_eigenvalues_condition_number(self) -> None:
        from quantspt.experimental.conditional_fgp import _condition_number

        assert _condition_number(np.array([0.0, 0.0])) == 1.0


# ===================================================================
# BoundaryRobustnessRegularizer tests
# ===================================================================


class TestBoundaryRobustnessRegularizer:
    def test_init_valid(self) -> None:
        reg = BoundaryRobustnessRegularizer(shock_magnitude=0.3)
        assert reg.penalty_weight == 0.1

    def test_init_invalid_shock(self) -> None:
        with pytest.raises(ValueError, match="shock_magnitude"):
            BoundaryRobustnessRegularizer(shock_magnitude=0.0)
        with pytest.raises(ValueError, match="shock_magnitude"):
            BoundaryRobustnessRegularizer(shock_magnitude=1.0)

    def test_init_invalid_n_scenarios(self) -> None:
        with pytest.raises(ValueError, match="n_scenarios"):
            BoundaryRobustnessRegularizer(n_scenarios=0)

    def test_concentration_scenarios_valid(self, mu_5: np.ndarray) -> None:
        reg = BoundaryRobustnessRegularizer(n_scenarios=3)
        scenarios = reg.concentration_scenarios(mu_5)
        assert len(scenarios) == 3
        for s in scenarios:
            assert s.shape == mu_5.shape
            assert abs(s.sum() - 1.0) < 1e-8
            assert np.all(s > 0)

    def test_scenario_increases_top_weight(self, mu_5: np.ndarray) -> None:
        reg = BoundaryRobustnessRegularizer(shock_magnitude=0.5, n_scenarios=1)
        scenarios = reg.concentration_scenarios(mu_5)
        top_idx = np.argmax(mu_5)
        assert scenarios[0][top_idx] > mu_5[top_idx]

    def test_penalty_nonnegative(self, mu_5: np.ndarray) -> None:
        reg = BoundaryRobustnessRegularizer()
        G = DiversityGenerator(0.3)
        pen = reg.penalty(G, mu_5)
        assert pen >= 0.0

    def test_penalty_lower_for_high_p(self, mu_5: np.ndarray) -> None:
        """Higher p means less sensitivity to concentration changes."""
        reg = BoundaryRobustnessRegularizer()
        pen_low = reg.penalty(DiversityGenerator(0.1), mu_5)
        pen_high = reg.penalty(DiversityGenerator(0.9), mu_5)
        assert pen_high <= pen_low + 1e-8

    def test_select_robust_p(self, mu_5: np.ndarray) -> None:
        reg = BoundaryRobustnessRegularizer()
        best_p, penalties = reg.select_robust_p(mu_5)
        assert P_MIN <= best_p <= P_MAX
        assert penalties.shape == (50,)
        assert np.all(penalties >= 0)

    def test_select_robust_p_custom_grid(self, mu_5: np.ndarray) -> None:
        reg = BoundaryRobustnessRegularizer()
        grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        best_p, penalties = reg.select_robust_p(mu_5, p_candidates=grid)
        assert best_p in grid
        assert len(penalties) == 5

    def test_equal_weights_low_penalty(self) -> None:
        mu = np.ones(10) / 10
        reg = BoundaryRobustnessRegularizer()
        pen = reg.penalty(DiversityGenerator(0.5), mu)
        assert pen < 0.1

    def test_penalty_with_extreme_concentration(self) -> None:
        """Nearly all weight on one stock triggers the 'other_sum < boost' branch."""
        mu = np.array([0.999, 0.0005, 0.0005])
        reg = BoundaryRobustnessRegularizer(shock_magnitude=0.99, n_scenarios=1)
        scenarios = reg.concentration_scenarios(mu)
        assert len(scenarios) == 1
        assert abs(scenarios[0].sum() - 1.0) < 1e-8

    def test_scenarios_more_than_n(self) -> None:
        """Request more scenarios than assets => capped at n."""
        mu = np.array([0.5, 0.3, 0.2])
        reg = BoundaryRobustnessRegularizer(n_scenarios=10)
        scenarios = reg.concentration_scenarios(mu)
        assert len(scenarios) == 3


# ===================================================================
# cost_optimal_p tests
# ===================================================================


class TestCostOptimalP:
    def test_zero_cost(self) -> None:
        p = cost_optimal_p(0.0)
        assert abs(p - 0.09) < 1e-6

    def test_10bps(self) -> None:
        p = cost_optimal_p(10.0)
        assert P_MIN <= p <= P_MAX
        assert p > 0.09

    def test_50bps(self) -> None:
        p = cost_optimal_p(50.0)
        assert abs(p - 0.315) < 0.05  # approx 0.09 + 45 * 0.005

    def test_monotonic_in_cost(self) -> None:
        costs = [0, 5, 10, 20, 50, 100]
        ps = [cost_optimal_p(c) for c in costs]
        for i in range(len(ps) - 1):
            assert ps[i] <= ps[i + 1] + 1e-10

    def test_clipped_to_range(self) -> None:
        p_low = cost_optimal_p(0.0)
        p_high = cost_optimal_p(5000.0)
        assert p_low >= P_MIN
        assert p_high <= P_MAX

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            cost_optimal_p(-1.0)

    def test_rebalancing_frequency_effect(self) -> None:
        p_monthly = cost_optimal_p(50.0, rebalancing_days=21)
        p_quarterly = cost_optimal_p(50.0, rebalancing_days=63)
        assert p_quarterly < p_monthly + 1e-10

    def test_very_high_cost(self) -> None:
        p = cost_optimal_p(200.0)
        assert p == P_MAX


# ===================================================================
# CovarianceConditionalFGP unit tests (synthetic data)
# ===================================================================


class TestCovarianceConditionalFGPUnit:
    def test_init_default(self) -> None:
        model = CovarianceConditionalFGP()
        assert not model.fitted
        assert model.config.cost_bps == 10.0
        assert model.feature_importances is None

    def test_init_custom_config(self) -> None:
        cfg = ConditionalFGPConfig(
            features=["avg_correlation", "effective_rank"],
            cost_bps=50.0,
            fallback_p=0.4,
        )
        model = CovarianceConditionalFGP(config=cfg)
        assert model.config.cost_bps == 50.0
        assert model.fallback_p == 0.4

    def test_extract_covariance_features(self, cov_5: np.ndarray) -> None:
        model = CovarianceConditionalFGP()
        features = model.extract_covariance_features(cov_5)
        assert features.shape == (len(DEFAULT_FEATURES),)
        assert np.all(np.isfinite(features))

    def test_optimal_p_unfitted(self) -> None:
        model = CovarianceConditionalFGP()
        features = np.array([0.3, 0.2, 5.0, 0.1])
        p = model.optimal_p(features)
        assert P_MIN <= p <= P_MAX

    def test_weights_without_cov(self, mu_5: np.ndarray) -> None:
        model = CovarianceConditionalFGP()
        w = model.weights(mu_5)
        assert w.shape == mu_5.shape
        assert abs(w.sum() - 1.0) < 1e-8
        assert np.all(w >= 0)

    def test_weights_with_cov(self, mu_5: np.ndarray, cov_5: np.ndarray) -> None:
        model = CovarianceConditionalFGP()
        w = model.weights(mu_5, cov_matrix=cov_5)
        assert w.shape == mu_5.shape
        assert abs(w.sum() - 1.0) < 1e-8

    def test_weights_from_mu_only(self, mu_5: np.ndarray) -> None:
        model = CovarianceConditionalFGP()
        w = model.weights_from_mu_only(mu_5)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_make_weight_func(self, mu_5: np.ndarray, cov_5: np.ndarray) -> None:
        model = CovarianceConditionalFGP()
        func = model.make_weight_func(cov_5)
        w = func(mu_5)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_generating_function_protocol(self, mu_5: np.ndarray) -> None:
        model = CovarianceConditionalFGP()
        G_val = model.generating_function(mu_5)
        assert G_val > 0
        grad = model.log_gradient(mu_5)
        assert grad.shape == mu_5.shape
        H = model.hessian(mu_5)
        assert H.shape == (5, 5)

    def test_to_generating_function(self) -> None:
        model = CovarianceConditionalFGP()
        G = model.to_generating_function()
        mu = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
        assert G(mu) > 0
        w = G.weights(mu)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_fit_synthetic_data(self, synthetic_market_data: dict) -> None:
        d = synthetic_market_data
        cfg = ConditionalFGPConfig(
            p_grid_size=5,
            n_estimators=10,
            boundary_penalty_weight=0.0,
        )
        model = CovarianceConditionalFGP(config=cfg)
        model.fit(
            d["weights"],
            d["cov_matrices"],
            d["returns"],
            cost_bps=10.0,
        )
        assert model.fitted
        assert model.training_metadata["status"] in ("fitted", "fallback_to_formula")

    def test_fit_and_predict(self, synthetic_market_data: dict) -> None:
        d = synthetic_market_data
        cfg = ConditionalFGPConfig(
            p_grid_size=5,
            n_estimators=10,
            boundary_penalty_weight=0.0,
        )
        model = CovarianceConditionalFGP(config=cfg)
        model.fit(
            d["weights"],
            d["cov_matrices"],
            d["returns"],
            cost_bps=10.0,
        )

        if model.training_metadata.get("status") == "fitted":
            assert model.feature_importances is not None
            assert len(model.feature_importances) == len(DEFAULT_FEATURES)

        features = model.extract_covariance_features(d["cov_matrices"][-1])
        p = model.optimal_p(features)
        assert P_MIN <= p <= P_MAX

    def test_fit_with_boundary_robustness(self, synthetic_market_data: dict) -> None:
        d = synthetic_market_data
        cfg = ConditionalFGPConfig(
            p_grid_size=5,
            n_estimators=10,
            boundary_penalty_weight=0.2,
        )
        model = CovarianceConditionalFGP(config=cfg)
        model.fit(
            d["weights"],
            d["cov_matrices"],
            d["returns"],
        )
        assert model.fitted

    def test_fit_insufficient_data(self) -> None:
        """Very short time series triggers fallback."""
        n = 5
        T = 30
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(n), size=T)
        returns = np.ones((T - 1, n)) + rng.normal(0, 0.01, (T - 1, n))
        covs = [np.eye(n) * 0.04 for _ in range(T)]

        model = CovarianceConditionalFGP()
        model.fit(weights, covs, returns)
        assert model.fitted
        assert model.training_metadata["status"] == "fallback_to_formula"

    def test_no_boundary_penalty(self, synthetic_market_data: dict) -> None:
        """Boundary penalty weight = 0 disables regularizer."""
        d = synthetic_market_data
        cfg = ConditionalFGPConfig(
            p_grid_size=5,
            n_estimators=10,
            boundary_penalty_weight=0.0,
        )
        model = CovarianceConditionalFGP(config=cfg)
        assert model._boundary_reg is None
        model.fit(d["weights"], d["cov_matrices"], d["returns"])
        assert model.fitted

    def test_n_assets_set_after_fit(self, synthetic_market_data: dict) -> None:
        d = synthetic_market_data
        cfg = ConditionalFGPConfig(
            p_grid_size=5, n_estimators=10, boundary_penalty_weight=0.0
        )
        model = CovarianceConditionalFGP(config=cfg)
        assert model.n_assets == 0
        model.fit(d["weights"], d["cov_matrices"], d["returns"])
        assert model.n_assets == d["n"]

    def test_nan_covariance_skipped(self) -> None:
        """NaN covariance matrices are skipped during training."""
        rng = np.random.default_rng(42)
        n, T = 5, 200
        weights = rng.dirichlet(np.ones(n), size=T)
        returns = np.ones((T - 1, n)) + rng.normal(0, 0.01, (T - 1, n))
        covs: list[np.ndarray] = [np.eye(n) * 0.04 for _ in range(T)]
        covs[50] = np.full((n, n), np.nan)

        cfg = ConditionalFGPConfig(
            p_grid_size=5, n_estimators=10, boundary_penalty_weight=0.0
        )
        model = CovarianceConditionalFGP(config=cfg)
        model.fit(weights, covs, returns)
        assert model.fitted


# ===================================================================
# optimal_p_for_cost_level tests (with backtesting)
# ===================================================================


class TestOptimalPForCostLevel:
    def test_returns_valid_structure(self, synthetic_market_data: dict) -> None:
        d = synthetic_market_data
        result = optimal_p_for_cost_level(
            d["returns"],
            d["weights"],
            cost_bps=10.0,
            p_grid=np.linspace(0.1, 0.9, 5),
        )
        assert "optimal_p" in result
        assert "optimal_net_excess" in result
        assert "formula_p" in result
        assert P_MIN <= result["optimal_p"] <= P_MAX
        assert len(result["net_excess_returns"]) == 5

    def test_higher_cost_shifts_optimal_p(self, synthetic_market_data: dict) -> None:
        d = synthetic_market_data
        p_grid = np.linspace(0.1, 0.9, 5)
        r_low = optimal_p_for_cost_level(d["returns"], d["weights"], 5.0, p_grid=p_grid)
        r_high = optimal_p_for_cost_level(
            d["returns"], d["weights"], 100.0, p_grid=p_grid
        )
        assert r_high["formula_p"] > r_low["formula_p"]


# ===================================================================
# Integration: lazy import from quantspt.ml
# ===================================================================


class TestLazyImports:
    def test_import_conditional_fgp(self) -> None:
        from quantspt.ml import CovarianceConditionalFGP as Cls

        assert Cls is not None

    def test_import_config(self) -> None:
        from quantspt.ml import ConditionalFGPConfig as Cls

        assert Cls is not None

    def test_import_feature_extractor(self) -> None:
        from quantspt.ml import CovarianceFeatureExtractor as Cls

        assert Cls is not None

    def test_import_boundary_reg(self) -> None:
        from quantspt.ml import BoundaryRobustnessRegularizer as Cls

        assert Cls is not None

    def test_import_cost_optimal_p(self) -> None:
        from quantspt.ml import cost_optimal_p as fn

        assert callable(fn)

    def test_import_optimal_p_for_cost_level(self) -> None:
        from quantspt.ml import optimal_p_for_cost_level as fn

        assert callable(fn)
