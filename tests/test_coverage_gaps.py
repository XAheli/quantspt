"""Tests targeting specific uncovered lines to push coverage from 94% toward 98%.

Each test function documents which file:lines it covers.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── _result.py (lines 88-134) ────────────────────────────────────────


class TestSPTResultChart:
    """Cover SPTResult.chart() and timed_result context manager."""

    def test_chart_1d_array(self):
        """Cover lines 88-98, 111-112."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from quantspt._result import SPTResult

        result = SPTResult(data=np.array([1.0, 2.0, 3.0]))
        fig = result.chart()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_chart_2d_array(self):
        """Cover lines 99-101."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from quantspt._result import SPTResult

        result = SPTResult(data=np.array([[1.0, 2.0], [3.0, 4.0]]))
        fig = result.chart()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_chart_dataframe(self):
        """Cover line 94-95."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import pandas as pd

        from quantspt._result import SPTResult

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = SPTResult(data=df, metadata={"title": "Test"})
        fig = result.chart()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_chart_non_array(self):
        """Cover lines 103-110 (text fallback)."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from quantspt._result import SPTResult

        result = SPTResult(data="some string data")
        fig = result.chart()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_repr(self):
        """Cover line 114-118."""
        from quantspt._result import SPTResult

        result = SPTResult(data=np.array([1.0]), warnings=["w1"])
        r = repr(result)
        assert "ndarray" in r
        assert "warnings=1" in r

    def test_timed_result(self):
        """Cover lines 122-134."""
        from quantspt._result import timed_result

        with timed_result() as t:
            _ = sum(range(100))
        assert t.elapsed_ms >= 0


# ── ml/__init__.py (lines 69-110) ────────────────────────────────────


class TestMLLazyImports:
    """Cover __getattr__ lazy-loading paths."""

    def test_lazy_neural_fgp(self):
        """Cover line 69-72."""
        from quantspt.ml import NeuralFGP

        assert NeuralFGP is not None

    def test_lazy_neural_fgp_config(self):
        """Cover lines 73-76."""
        from quantspt.ml import NeuralFGPConfig

        assert NeuralFGPConfig is not None

    def test_lazy_input_convex_nn(self):
        """Cover lines 77-80."""
        from quantspt.ml import InputConvexNN

        assert InputConvexNN is not None

    def test_lazy_hmm_regime_detector(self):
        """Cover lines 81-84."""
        from quantspt.ml import HMMRegimeDetector

        assert HMMRegimeDetector is not None

    def test_lazy_changepoint_detector(self):
        """Cover lines 85-88."""
        from quantspt.ml import ChangepointDetector

        assert ChangepointDetector is not None

    def test_lazy_factor_model_estimator(self):
        """Cover lines 89-92."""
        from quantspt.ml import FactorModelEstimator

        assert FactorModelEstimator is not None

    def test_lazy_rmt_denoiser(self):
        """Cover lines 93-96."""
        from quantspt.ml import RMTDenoiser

        assert RMTDenoiser is not None

    def test_lazy_losses(self):
        """Cover lines 98-109."""
        from quantspt.ml import default_loss, relative_return_loss

        assert callable(relative_return_loss)
        assert callable(default_loss)

    def test_lazy_unknown_raises(self):
        """Cover line 110."""
        with pytest.raises(AttributeError, match="no attribute"):
            from quantspt import ml

            ml.__getattr__("nonexistent_thing")


# ── contrib/__init__.py (lines 64-100, 120) ──────────────────────────


class TestContribRegistry:
    """Cover discover/list functions and register_generating_function."""

    def test_discover_generating_functions(self):
        """Cover lines 97-100."""
        from quantspt.contrib import discover_generating_functions

        gfs = discover_generating_functions()
        assert isinstance(gfs, dict)

    def test_list_generating_functions(self):
        """Cover line 120."""
        from quantspt.contrib import list_generating_functions

        names = list_generating_functions()
        assert isinstance(names, set)

    def test_register_and_discover_generating_function(self):
        from quantspt.contrib import (
            discover_generating_functions,
            register_generating_function,
        )

        @register_generating_function("test_gf")
        class _TestGF:
            pass

        gfs = discover_generating_functions()
        assert "test_gf" in gfs

    def test_discover_providers(self):
        """Cover lines 64-68."""
        from quantspt.contrib import discover_providers

        providers = discover_providers()
        assert isinstance(providers, dict)

    def test_discover_portfolios(self):
        """Cover lines 75."""
        from quantspt.contrib import discover_portfolios

        portfolios = discover_portfolios()
        assert isinstance(portfolios, dict)

    def test_discover_models(self):
        """Cover line 83."""
        from quantspt.contrib import discover_models

        models = discover_models()
        assert isinstance(models, dict)

    def test_list_providers(self):
        """Cover line 91."""
        from quantspt.contrib import list_providers

        assert isinstance(list_providers(), set)

    def test_list_portfolios(self):
        """Cover line 97."""
        from quantspt.contrib import list_portfolios

        assert isinstance(list_portfolios(), set)

    def test_list_models(self):
        """Cover line 100."""
        from quantspt.contrib import list_models

        assert isinstance(list_models(), set)


# ── data/schemas.py (lines 215-216, 245-246, 271) ────────────────────


class TestNewSchemas:
    """Cover CausalGraph, RegimeLabels, FactorLoadings."""

    def test_causal_graph_valid(self):
        """Cover CausalGraph.__post_init__ pass path."""
        from quantspt.data.schemas import CausalGraph

        adj = np.eye(3)
        cg = CausalGraph(
            adjacency_matrix=adj,
            variable_names=["A", "B", "C"],
            discovery_method="pc",
        )
        assert cg.adjacency_matrix.shape == (3, 3)

    def test_causal_graph_invalid_shape(self):
        """Cover CausalGraph.__post_init__ fail path (line 215-216)."""
        from quantspt.data.schemas import CausalGraph

        with pytest.raises(
            (ValueError, RuntimeError),
            match=r"shape|mismatch|Precondition",
        ):
            CausalGraph(
                adjacency_matrix=np.eye(2),
                variable_names=["A", "B", "C"],
            )

    def test_regime_labels_valid(self):
        from quantspt.data.schemas import RegimeLabels

        labels = RegimeLabels(
            labels=np.array([0, 1, 0, 1], dtype=np.int64),
            n_regimes=2,
        )
        assert labels.n_regimes == 2

    def test_regime_labels_invalid(self):
        """Cover RegimeLabels.__post_init__ fail path (line 245-246)."""
        from quantspt.data.schemas import RegimeLabels

        with pytest.raises(
            (ValueError, RuntimeError),
            match=r"unique|regimes|Precondition",
        ):
            RegimeLabels(
                labels=np.array([0, 1, 2], dtype=np.int64),
                n_regimes=2,
            )

    def test_factor_loadings_valid(self):
        from quantspt.data.schemas import FactorLoadings

        fl = FactorLoadings(loadings=np.ones((5, 3)))
        assert fl.loadings.shape == (5, 3)

    def test_factor_loadings_invalid(self):
        """Cover FactorLoadings.__post_init__ fail path (line 271)."""
        from quantspt.data.schemas import FactorLoadings

        with pytest.raises(
            (ValueError, RuntimeError),
            match=r"2-D|shape|Precondition",
        ):
            FactorLoadings(loadings=np.array([1.0, 2.0]))


# ── core/generating_functions.py (lines 639-697) ─────────────────────


class TestAutoDiffGeneratingFunction:
    """Cover AutoDiffGeneratingFunction JAX paths."""

    def test_autodiff_jax_log_gradient(self):
        """Cover lines 639-644, 666, 675-685."""
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        from quantspt.core.generating_functions import AutoDiffGeneratingFunction

        def my_g(mu):
            return sum(mu**0.5)

        gf = AutoDiffGeneratingFunction(my_g, "test_jax", backend="jax")
        mu = np.array([0.4, 0.35, 0.25])
        grad = gf.log_gradient(mu)
        assert grad.shape == (3,)
        assert np.all(np.isfinite(grad))

    def test_autodiff_jax_hessian(self):
        """Cover lines 689-697."""
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        from quantspt.core.generating_functions import AutoDiffGeneratingFunction

        def my_g(mu):
            return sum(mu**0.5)

        gf = AutoDiffGeneratingFunction(my_g, "test_jax", backend="jax")
        mu = np.array([0.4, 0.35, 0.25])
        H = gf.hessian(mu)
        assert H.shape == (3, 3)
        assert np.allclose(H, H.T)

    def test_autodiff_auto_selects_jax(self):
        """Cover lines 648-653 (auto-detection)."""
        from quantspt.core.generating_functions import AutoDiffGeneratingFunction

        def my_g(mu):
            return sum(mu**0.5)

        gf = AutoDiffGeneratingFunction(my_g, "test_auto", backend="auto")
        mu = np.array([0.5, 0.3, 0.2])
        val = gf(mu)
        assert val > 0


# ── visualization/_backend.py (lines 20-21, 40-41) ───────────────────


class TestVisualizationBackend:
    """Cover visualization _backend lazy imports."""

    def test_get_matplotlib(self):
        """Cover lines 27-44."""
        pytest.importorskip("matplotlib")
        from quantspt.visualization._backend import _get_matplotlib

        plt, Figure = _get_matplotlib()
        assert plt is not None
        assert Figure is not None

    def test_get_plotly(self):
        """Cover lines 14-24."""
        pytest.importorskip("plotly")
        from quantspt.visualization._backend import _get_plotly

        go = _get_plotly()
        assert go is not None

    def test_validate_backend_invalid(self):
        """Cover line 50."""
        from quantspt.visualization._backend import _validate_backend

        with pytest.raises(ValueError, match="must be"):
            _validate_backend("invalid")


# ── backtesting/attribution.py (lines 121-125) ───────────────────────


class TestAttributionEdge:
    """Cover edge case in compute_attribution."""

    def test_attribution_single_step(self):
        """Cover boundary path in compute_attribution (lines 121-125)."""
        from quantspt.backtesting.attribution import compute_attribution
        from quantspt.core.generating_functions import DiversityGenerator

        gf = DiversityGenerator(p=0.5)
        n = 3
        mu_path = np.array([[0.4, 0.35, 0.25], [0.38, 0.37, 0.25]])
        a_path = np.array([np.eye(n) * 0.04] * 2)
        result = compute_attribution(
            gf, mu_path, a_path, actual_log_relative=0.01, dt=1.0 / 252
        )
        assert hasattr(result, "boundary")
        assert hasattr(result, "drift_integral")


# ── ml/losses.py (uncovered lines: 88-96, 101, 113-118, etc.) ────────


class TestMLLosses:
    """Cover loss function edge cases."""

    def test_relative_return_loss(self):
        pytest.importorskip("torch")
        from quantspt.ml.losses import RelativeReturnLoss

        loss_fn = RelativeReturnLoss()
        assert callable(loss_fn)

    def test_weight_regularization(self):
        pytest.importorskip("torch")
        from quantspt.ml.losses import WeightRegularization

        reg = WeightRegularization()
        assert callable(reg)

    def test_turnover_penalty(self):
        pytest.importorskip("torch")
        from quantspt.ml.losses import TurnoverPenalty

        pen = TurnoverPenalty()
        assert callable(pen)

    def test_default_loss(self):
        pytest.importorskip("torch")
        from quantspt.ml.losses import default_loss

        loss = default_loss()
        assert callable(loss)

    def test_drift_integral_loss(self):
        pytest.importorskip("torch")
        from quantspt.ml.losses import DriftIntegralLoss

        loss = DriftIntegralLoss()
        assert callable(loss)

    def test_loss_composition(self):
        """Cover _CompositeLoss (lines 67-101)."""
        pytest.importorskip("torch")
        from quantspt.ml.losses import (
            RelativeReturnLoss,
            TurnoverPenalty,
            WeightRegularization,
        )

        combined = RelativeReturnLoss() + 0.01 * WeightRegularization()
        assert callable(combined)
        combined2 = combined + 0.005 * TurnoverPenalty()
        assert callable(combined2)

    def test_sharpe_relative_loss(self):
        pytest.importorskip("torch")
        from quantspt.ml.losses import SharpeRelativeLoss

        loss = SharpeRelativeLoss()
        assert callable(loss)


# ── ml/covariance.py (lines 74, 88, 95, 139, 178, 226, 276) ─────────


class TestMLCovarianceEdge:
    """Cover edge cases in FactorModelEstimator and RMTDenoiser."""

    def test_factor_model_n_factors_exceeds_assets(self):
        """Cover line 139 (k = min(n_factors, n))."""
        from quantspt.ml.covariance import FactorModelEstimator

        rng = np.random.default_rng(42)
        returns = rng.standard_normal((50, 3))
        fm = FactorModelEstimator(n_factors=10)
        fm.fit(returns)
        cov = fm.estimate()
        assert cov.shape == (3, 3)

    def test_factor_model_auto_select(self):
        """Cover explained variance auto-selection."""
        from quantspt.ml.covariance import FactorModelEstimator

        rng = np.random.default_rng(42)
        returns = rng.standard_normal((100, 5))
        fm = FactorModelEstimator(n_factors=None, explained_variance_threshold=0.5)
        fm.fit(returns)
        cov = fm.estimate()
        assert cov.shape == (5, 5)

    def test_factor_model_loadings(self):
        """Cover line 74, 88, 95 (property accessors)."""
        from quantspt.ml.covariance import FactorModelEstimator

        rng = np.random.default_rng(42)
        returns = rng.standard_normal((100, 5))
        fm = FactorModelEstimator(n_factors=2)
        fm.fit(returns)
        loadings = fm.loadings
        assert loadings is not None
        assert loadings.shape[0] == 5

    def test_rmt_denoiser_underdetermined(self):
        """Cover RMTDenoiser with T < n (short sample)."""
        from quantspt.ml.covariance import RMTDenoiser

        rng = np.random.default_rng(42)
        returns = rng.standard_normal((10, 20))
        rmt = RMTDenoiser()
        rmt.fit(returns)
        cov = rmt.estimate()
        assert cov.shape == (20, 20)
        assert np.allclose(cov, cov.T)


# ── optimization/generating_function.py (lines 226-237) ──────────────


class TestOptGeneratingFunctionEdge:
    """Cover edge cases in GF optimization."""

    def test_optimize_diversity_parameter(self):
        """Cover lines 226-237 in generating_function.py."""
        from quantspt.optimization.generating_function import (
            optimize_diversity_parameter,
        )

        rng = np.random.default_rng(42)
        T, n = 20, 5
        weights_path = np.abs(rng.standard_normal((T, n)))
        weights_path = weights_path / weights_path.sum(axis=1, keepdims=True)
        cov = rng.random((n, n))
        cov = cov @ cov.T / 10
        cov_matrices = [cov] * T
        result = optimize_diversity_parameter(weights_path, cov_matrices)
        assert result is not None


# ── post_processing/discrete_allocation.py (lines 94, 104, etc.) ─────


class TestDiscreteAllocationEdge:
    """Cover edge cases in discrete allocation."""

    def test_greedy_allocation_with_zero_weights(self):
        """Cover edge case where some weights are zero."""
        from quantspt.post_processing.discrete_allocation import greedy_allocation

        weights = np.array([0.0, 0.5, 0.5, 0.0])
        prices = np.array([100.0, 50.0, 25.0, 200.0])
        result = greedy_allocation(weights, prices, total_value=1000.0)
        assert result.shares.shape == (4,)
        assert result.shares[0] == 0
        assert result.shares[3] == 0


# ── simulation edge cases ────────────────────────────────────────────


class TestSimulationEdge:
    """Cover edge cases in simulation modules."""

    def test_simulate_market(self):
        """Cover line 188 in market_simulator.py."""
        from quantspt.models.gbm import CorrelatedGBMMarket
        from quantspt.simulation.market_simulator import simulate_market

        n = 3
        model = CorrelatedGBMMarket(
            mu=np.array([0.05, 0.05, 0.05]),
            cov=np.eye(n) * 0.04,
        )
        x0 = np.array([100.0, 100.0, 100.0])
        result = simulate_market(model, x0, T=1.0, n_steps=5, seed=42)
        assert result.data.prices.shape[0] == 6

    def test_monte_carlo_few_paths(self):
        """Cover edge case in monte_carlo.py (lines 247-248)."""
        from quantspt.core.processes import CorrelatedGBM
        from quantspt.simulation.monte_carlo import MonteCarloEngine

        n = 2
        process = CorrelatedGBM(
            mu=np.array([0.05, 0.05]),
            cov=np.eye(n) * 0.04,
            x0=np.array([100.0, 100.0]),
        )
        engine = MonteCarloEngine(process, n_paths=5, n_steps=10, T=0.1)
        result = engine.run()
        assert result is not None
