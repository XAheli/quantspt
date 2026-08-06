"""Tests targeting every remaining coverage gap across the codebase.

Systematically exercises uncovered branches in: JAX backend, visualization
backend, losses, neural FGP, regime, export, covariance, csv_parquet,
discrete allocation, protocols, and optimization modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# JAX Backend: import guard, simulate_gbm_step, covariance_shrinkage, name
# ---------------------------------------------------------------------------


class TestJaxBackendCoverage:
    """Cover remaining JAX backend branches."""

    def test_require_jax_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from quantspt._backends.jax_backend import _require_jax

        monkeypatch.setitem(sys.modules, "jax", None)
        with pytest.raises(ImportError, match="quantspt\\[gpu\\]"):
            _require_jax()

    def test_name_property(self) -> None:
        jax_mod = pytest.importorskip("jax")
        from quantspt._backends.jax_backend import JaxBackend

        jax_mod.config.update("jax_enable_x64", True)
        backend = JaxBackend()
        assert backend.name == "jax"

    def test_simulate_gbm_step(self) -> None:
        jax_mod = pytest.importorskip("jax")
        from quantspt._backends.jax_backend import JaxBackend

        jax_mod.config.update("jax_enable_x64", True)
        backend = JaxBackend()
        rng = np.random.default_rng(42)
        n = 3
        x = np.ones(n) * 100.0
        mu = np.ones(n) * 0.05
        chol = np.eye(n) * 0.2
        dt = 1 / 252
        dw = rng.standard_normal(n) * np.sqrt(dt)
        result = backend.simulate_gbm_step(x, mu, chol, dt, dw)
        assert result.shape == (n,)
        assert np.all(result > 0)
        assert result.dtype == np.float64

    def test_covariance_shrinkage(self) -> None:
        jax_mod = pytest.importorskip("jax")
        from quantspt._backends.jax_backend import JaxBackend
        from quantspt._backends.numpy_backend import NumpyBackend

        jax_mod.config.update("jax_enable_x64", True)
        jb = JaxBackend()
        npb = NumpyBackend()
        rng = np.random.default_rng(42)
        returns = rng.standard_normal((50, 5))
        jax_result = jb.covariance_shrinkage(returns, 0.3)
        np_result = npb.covariance_shrinkage(returns, 0.3)
        np.testing.assert_allclose(jax_result, np_result, atol=1e-8)

    def test_gradient(self) -> None:
        jax_mod = pytest.importorskip("jax")
        import jax.numpy as jnp

        from quantspt._backends.jax_backend import JaxBackend

        jax_mod.config.update("jax_enable_x64", True)
        backend = JaxBackend()

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x**2)

        x = np.array([1.0, 2.0, 3.0])
        grad = backend.gradient(f, x)
        np.testing.assert_allclose(grad, 2.0 * x, atol=1e-10)

    def test_hessian(self) -> None:
        jax_mod = pytest.importorskip("jax")
        import jax.numpy as jnp

        from quantspt._backends.jax_backend import JaxBackend

        jax_mod.config.update("jax_enable_x64", True)
        backend = JaxBackend()

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x**2)

        x = np.array([1.0, 2.0, 3.0])
        H = backend.hessian(f, x)
        np.testing.assert_allclose(H, 2.0 * np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# Visualization _backend: import error paths
# ---------------------------------------------------------------------------


class TestVisualizationBackend:
    """Cover import-error branches in visualization._backend."""

    def test_plotly_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from quantspt.visualization._backend import _get_plotly

        monkeypatch.setitem(sys.modules, "plotly", None)
        monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
        with pytest.raises(ImportError, match="plotly"):
            _get_plotly()

    def test_matplotlib_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from quantspt.visualization._backend import _get_matplotlib

        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        with pytest.raises(ImportError, match="matplotlib"):
            _get_matplotlib()


# ---------------------------------------------------------------------------
# ml/losses.py: _CompositeLoss and _BaseLoss arithmetic edge cases
# ---------------------------------------------------------------------------


class TestLossesCoverage:
    """Cover remaining loss function branches."""

    def test_composite_radd_with_composite(self) -> None:
        from quantspt.ml.losses import relative_return_loss, turnover_penalty

        c1 = relative_return_loss + turnover_penalty
        c2 = relative_return_loss + turnover_penalty
        combined = c1.__radd__(c2)
        assert len(combined.terms) == 4

    def test_composite_radd_with_base(self) -> None:
        from quantspt.ml.losses import relative_return_loss, turnover_penalty

        composite = relative_return_loss + turnover_penalty
        combined = composite.__radd__(relative_return_loss)
        assert len(combined.terms) == 3

    def test_composite_add_with_composite(self) -> None:
        from quantspt.ml.losses import relative_return_loss, turnover_penalty

        c1 = relative_return_loss + turnover_penalty
        c2 = relative_return_loss + turnover_penalty
        combined = c1 + c2
        assert len(combined.terms) == 4

    def test_base_radd_with_composite(self) -> None:
        from quantspt.ml.losses import relative_return_loss, turnover_penalty

        composite = relative_return_loss + turnover_penalty
        combined = turnover_penalty.__radd__(composite)
        assert len(combined.terms) == 3

    def test_base_add_with_composite(self) -> None:
        from quantspt.ml.losses import relative_return_loss, turnover_penalty

        composite = relative_return_loss + turnover_penalty
        combined = turnover_penalty.__add__(composite)
        assert len(combined.terms) == 3

    def test_base_rmul(self) -> None:
        from quantspt.ml.losses import turnover_penalty

        scaled = turnover_penalty.__rmul__(0.5)
        assert len(scaled.terms) == 1
        assert scaled.terms[0][0] == 0.5

    def test_base_mul(self) -> None:
        from quantspt.ml.losses import turnover_penalty

        scaled = turnover_penalty.__mul__(2.0)
        assert len(scaled.terms) == 1
        assert scaled.terms[0][0] == 2.0

    def test_composite_rmul(self) -> None:
        from quantspt.ml.losses import relative_return_loss, turnover_penalty

        composite = relative_return_loss + turnover_penalty
        scaled = 0.5 * composite
        assert all(abs(c) <= 0.5 + 1e-10 for c, _ in scaled.terms)

    def test_turnover_penalty_single_timestep(self) -> None:
        from quantspt.ml.losses import turnover_penalty

        weights = torch.rand(1, 5)
        returns = 1.0 + torch.randn(1, 5) * 0.01
        loss = turnover_penalty(weights, returns)
        assert loss.item() == pytest.approx(0.0)

    def test_drift_integral_loss_full(self) -> None:
        from quantspt.ml.losses import DriftIntegralLoss, drift_integral_loss

        T, n = 20, 5
        rng = np.random.default_rng(42)
        alpha = rng.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)
        pw = alpha / alpha.sum(axis=1, keepdims=True) * 0.9 + 0.1 / n
        returns = 1.0 + rng.standard_normal((T, n)) * 0.01
        L = rng.standard_normal((n, n)) * 0.1
        covs = np.array([L @ L.T + np.eye(n) * 0.01 for _ in range(T)])

        loss_val = drift_integral_loss(pw, mw, covs, 1 / 252)
        assert torch.isfinite(loss_val)

        dil = DriftIntegralLoss(dt=1 / 252)
        pw_t = torch.tensor(pw, dtype=torch.float64)
        returns_t = torch.tensor(returns, dtype=torch.float64)
        loss = dil(
            pw_t,
            returns_t,
            market_weights=mw,
            covariance_matrices=covs,
        )
        assert torch.isfinite(loss)

    def test_drift_integral_loss_missing_args(self) -> None:
        from quantspt.ml.losses import DriftIntegralLoss

        dil = DriftIntegralLoss()
        w = torch.rand(10, 5)
        r = torch.rand(10, 5)
        with pytest.raises(ValueError, match="market_weights"):
            dil(w, r)

    def test_sharpe_relative_loss(self) -> None:
        from quantspt.ml.losses import sharpe_of_relative_loss

        weights = torch.rand(20, 5)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        returns = 1.0 + torch.randn(20, 5) * 0.01
        loss = sharpe_of_relative_loss(weights, returns)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# ml/regime.py: import guards, predict_proba 1d, forecast_diversity
# ---------------------------------------------------------------------------


class TestRegimeCoverage:
    """Cover remaining regime detection branches."""

    def test_require_hmmlearn_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from quantspt.ml.regime import _require_hmmlearn

        monkeypatch.setitem(sys.modules, "hmmlearn", None)
        with pytest.raises(ImportError, match="quantspt\\[ml\\]"):
            _require_hmmlearn()

    def test_require_ruptures_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from quantspt.ml.regime import _require_ruptures

        monkeypatch.setitem(sys.modules, "ruptures", None)
        with pytest.raises(ImportError, match="quantspt\\[ml\\]"):
            _require_ruptures()

    def test_hmm_fit_with_n_regimes_override(self) -> None:
        from quantspt.ml.regime import HMMRegimeDetector

        rng = np.random.default_rng(42)
        features = np.concatenate(
            [rng.normal(0, 1, (100, 2)), rng.normal(3, 1, (100, 2))]
        )
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(features, n_regimes=3)
        assert detector.n_regimes == 3

    def test_hmm_predict_proba_1d(self) -> None:
        from quantspt.ml.regime import HMMRegimeDetector

        rng = np.random.default_rng(42)
        features = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(features)
        proba = detector.predict_proba(features)
        assert proba.shape == (200, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_hmm_forecast_diversity(self) -> None:
        from quantspt.ml.regime import HMMRegimeDetector

        rng = np.random.default_rng(42)
        features = np.concatenate(
            [rng.normal(0, 1, (100, 2)), rng.normal(3, 1, (100, 2))]
        )
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(features)
        forecast = detector.forecast_diversity(horizon=10)
        assert forecast.shape == (10, 2)
        np.testing.assert_allclose(forecast.sum(axis=1), 1.0, atol=1e-10)

    def test_changepoint_predict_2d(self) -> None:
        from quantspt.ml.regime import ChangepointDetector

        rng = np.random.default_rng(42)
        signal = np.concatenate(
            [rng.normal(0, 0.1, (80, 2)), rng.normal(5, 0.1, (80, 2))]
        )
        detector = ChangepointDetector(penalty=1.0, min_size=10)
        labels = detector.predict(signal)
        assert labels.shape == (160,)
        assert labels.dtype == np.int64

    def test_changepoint_predict_unfitted(self) -> None:
        from quantspt.ml.regime import ChangepointDetector

        rng = np.random.default_rng(42)
        signal = np.concatenate([rng.normal(0, 0.1, 50), rng.normal(5, 0.1, 50)])
        detector = ChangepointDetector(penalty=1.0, min_size=10)
        labels = detector.predict(signal)
        assert labels.shape == (100,)


# ---------------------------------------------------------------------------
# ml/neural_fgp.py: import guard
# ---------------------------------------------------------------------------


class TestNeuralFGPCoverage:
    """Cover neural FGP import guard."""

    def test_require_torch_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from quantspt.ml.neural_fgp import _require_torch

        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError, match="quantspt\\[ml\\]"):
            _require_torch()


# ---------------------------------------------------------------------------
# ml/covariance.py: unfitted property access
# ---------------------------------------------------------------------------


class TestCovarianceCoverage:
    """Cover remaining covariance module branches."""

    def test_factor_model_unfitted_properties(self) -> None:
        from quantspt.ml.covariance import FactorModelEstimator

        model = FactorModelEstimator(n_factors=3)
        with pytest.raises(RuntimeError, match="fitted"):
            _ = model.n_factors
        with pytest.raises(RuntimeError, match="fitted"):
            _ = model.loadings
        with pytest.raises(RuntimeError, match="fitted"):
            _ = model.factor_covariance
        with pytest.raises(RuntimeError, match="fitted"):
            _ = model.idiosyncratic_variance

    def test_rmt_unfitted_estimate(self) -> None:
        from quantspt.ml.covariance import RMTDenoiser

        model = RMTDenoiser()
        assert model.n_assets == 0
        with pytest.raises(RuntimeError, match="fitted"):
            model.estimate()

    def test_rmt_invalid_method(self) -> None:
        from quantspt.ml.covariance import RMTDenoiser

        with pytest.raises(ValueError, match="method must be"):
            RMTDenoiser(method="bogus")


# ---------------------------------------------------------------------------
# ml/_protocols.py: LearnedGeneratingFunction validation
# ---------------------------------------------------------------------------


class TestProtocolsCoverage:
    """Cover LearnedGeneratingFunction validation branches."""

    def test_learned_gf_validation_hessian_check(self) -> None:
        from quantspt.errors import SPTInvariantError
        from quantspt.ml._protocols import LearnedGeneratingFunction

        class BadModel:
            def generating_function(self, mu: np.ndarray) -> float:
                return float(np.sum(mu**0.5))

            def log_gradient(self, mu: np.ndarray) -> np.ndarray:
                return np.zeros_like(mu)

            def hessian(self, mu: np.ndarray) -> np.ndarray:
                return np.eye(len(mu)) * 10.0

            def to_generating_function(self):  # type: ignore[no-untyped-def]
                pass

        with pytest.raises(SPTInvariantError, match="negative semi-definite"):
            LearnedGeneratingFunction(
                BadModel(),
                name_str="Bad",
                n_assets=5,
                skip_validation=False,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# visualization/export.py: PDF, LaTeX, report branches
# ---------------------------------------------------------------------------


class TestExportCoverage:
    """Cover remaining export format branches."""

    def test_to_latex_pdf(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        from quantspt.visualization.export import to_latex

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        out = to_latex(fig, tmp_path / "test.pdf")
        assert out.exists()

    def test_to_latex_pgf(self, tmp_path: Path) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        from quantspt.visualization.export import to_latex

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        out = to_latex(fig, tmp_path / "test.png")
        assert out.exists()

    def test_to_pdf_matplotlib(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        from quantspt.visualization.export import to_pdf

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        out = to_pdf(fig, tmp_path / "test.pdf")
        assert out.exists()

    def test_to_html_matplotlib(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        from quantspt.visualization.export import to_html

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        out = to_html(fig, tmp_path / "test.html")
        assert out.exists()
        content = out.read_text()
        assert "<svg" in content

    def test_to_html_matplotlib_fragment(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        from quantspt.visualization.export import to_html

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        out = to_html(fig, tmp_path / "test.html", full_html=False)
        assert out.exists()

    def test_generate_report_pdf(self, tmp_path: Path) -> None:
        from quantspt.visualization.export import generate_report

        result = {
            "returns": np.random.default_rng(0).standard_normal(50) * 0.01,
            "metrics": {"sharpe": 1.5, "max_drawdown": 0.1, "total_return": 0.25},
        }
        out = generate_report(result, tmp_path / "report.pdf", format="pdf")
        assert out.exists()

    def test_generate_report_latex(self, tmp_path: Path) -> None:
        from quantspt.visualization.export import generate_report

        result = {
            "returns": np.random.default_rng(0).standard_normal(50) * 0.01,
            "metrics": {"sharpe": 1.5, "total_return": "0.25"},
        }
        out = generate_report(result, tmp_path / "report.tex", format="latex")
        assert out.exists()
        content = out.read_text()
        assert "\\begin{document}" in content

    def test_generate_report_latex_without_metrics(self, tmp_path: Path) -> None:
        from quantspt.visualization.export import generate_report

        result = {"returns": np.random.default_rng(0).standard_normal(50) * 0.01}
        out = generate_report(result, tmp_path / "report.tex", format="latex")
        assert out.exists()

    def test_generate_report_html_with_weights(self, tmp_path: Path) -> None:
        import pandas as pd

        from quantspt.visualization.export import generate_report

        result = {
            "returns": np.random.default_rng(0).standard_normal(50) * 0.01,
            "metrics": {"sharpe": 1.5},
            "weights": pd.DataFrame(
                {"AAPL": [0.3, 0.4], "MSFT": [0.7, 0.6]},
                index=pd.date_range("2020-01-01", periods=2),
            ),
        }
        out = generate_report(result, tmp_path / "report.html", format="html")
        assert out.exists()
        content = out.read_text()
        assert "AAPL" in content

    def test_render_weights_dict(self, tmp_path: Path) -> None:
        from quantspt.visualization.export import _render_weights_html

        html = _render_weights_html({"AAPL": 0.5, "MSFT": 0.5})
        assert "AAPL" in html

    def test_render_weights_unsupported(self) -> None:
        from quantspt.visualization.export import _render_weights_html

        result = _render_weights_html([0.5, 0.5])
        assert result == ""

    def test_render_metrics_non_float(self) -> None:
        from quantspt.visualization.export import _render_metrics_html

        html = _render_metrics_html({"status": "good", "sharpe": 1.5})
        assert "good" in html

    def test_normalize_result_object(self) -> None:
        from quantspt.visualization.export import _normalize_result

        class FakeResult:
            def __init__(self) -> None:
                self.returns = [0.01, 0.02]
                self.metrics = {"sharpe": 1.0}

        result = _normalize_result(FakeResult())
        assert "returns" in result
        assert "metrics" in result

    def test_require_matplotlib_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from quantspt.visualization.export import _require_matplotlib

        monkeypatch.setitem(sys.modules, "matplotlib", None)
        with pytest.raises(ImportError, match="matplotlib"):
            _require_matplotlib()

    def test_require_plotly_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from quantspt.visualization.export import _require_plotly

        monkeypatch.setitem(sys.modules, "plotly", None)
        with pytest.raises(ImportError, match="plotly"):
            _require_plotly()


# ---------------------------------------------------------------------------
# data/providers/csv_parquet.py: edge cases
# ---------------------------------------------------------------------------


class TestCSVParquetCoverage:
    """Cover edge cases in CSV/Parquet providers."""

    def test_csv_detect_datetime_column(self, tmp_path: Path) -> None:
        import pandas as pd

        from quantspt.data.providers.csv_parquet import CSVProvider

        df = pd.DataFrame(
            {
                "weird_col": pd.date_range("2020-01-01", periods=5),
                "AAPL": [100.0, 101.0, 102.0, 103.0, 104.0],
                "MSFT": [200.0, 201.0, 202.0, 203.0, 204.0],
            }
        )
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        provider = CSVProvider(csv_path)
        result = provider.load(tickers=["AAPL", "MSFT"])
        assert result.data is not None

    def test_csv_no_date_column_uses_index(self, tmp_path: Path) -> None:
        import pandas as pd

        from quantspt.data.providers.csv_parquet import CSVProvider

        df = pd.DataFrame(
            {"AAPL": [100.0, 101.0], "MSFT": [200.0, 201.0]},
            index=pd.date_range("2020-01-01", periods=2),
        )
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path)
        provider = CSVProvider(csv_path)
        result = provider.load()
        assert result.data is not None

    def test_parquet_filter_by_tickers(self, tmp_path: Path) -> None:
        import pandas as pd

        from quantspt.data.providers.csv_parquet import ParquetProvider

        df = pd.DataFrame(
            {"AAPL": [100.0, 101.0], "MSFT": [200.0, 201.0], "GOOG": [300.0, 301.0]},
            index=pd.date_range("2020-01-01", periods=2),
        )
        pq_path = tmp_path / "test.parquet"
        df.to_parquet(pq_path)
        provider = ParquetProvider(pq_path)
        result = provider.load(tickers=["AAPL"])
        assert "AAPL" in result.data.tickers


# ---------------------------------------------------------------------------
# post_processing/discrete_allocation.py
# ---------------------------------------------------------------------------


class TestDiscreteAllocationCoverage:
    """Cover edge cases in discrete allocation."""

    def test_greedy_zero_invested(self) -> None:
        """When prices are too high for budget, get zero allocation."""
        from quantspt.post_processing.discrete_allocation import greedy_allocation

        result = greedy_allocation(
            weights=np.array([0.6, 0.4]),
            prices=np.array([10000.0, 20000.0]),
            total_value=1.0,
        )
        assert np.all(result.shares == 0)
        assert np.allclose(result.actual_weights, 0.0)

    def test_lp_allocation_basic(self) -> None:
        """LP allocation works for basic case."""
        from quantspt.post_processing.discrete_allocation import lp_allocation

        result = lp_allocation(
            weights=np.array([0.6, 0.4]),
            prices=np.array([100.0, 50.0]),
            total_value=10000.0,
        )
        assert np.all(result.shares >= 0)
        assert result.leftover_cash >= 0

    def test_lp_allocation_zero_actual(self) -> None:
        """LP allocation with very expensive assets gives zero weights."""
        from quantspt.post_processing.discrete_allocation import lp_allocation

        result = lp_allocation(
            weights=np.array([0.6, 0.4]),
            prices=np.array([100000.0, 200000.0]),
            total_value=100.0,
        )
        assert np.all(result.shares == 0)
        assert np.allclose(result.actual_weights, 0.0)


# ---------------------------------------------------------------------------
# optimization/generating_function.py: grid search exception + refinement
# ---------------------------------------------------------------------------


class TestOptimizationCoverage:
    """Cover remaining optimization branches."""

    def test_diversity_parameter_with_refinement(self) -> None:
        from quantspt.optimization.generating_function import (
            optimize_diversity_parameter,
        )

        rng = np.random.default_rng(42)
        n = 5
        alpha = rng.exponential(size=(50, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)
        L = rng.standard_normal((n, n)) * 0.1
        cov = L @ L.T + np.eye(n) * 0.01
        covs = np.array([cov for _ in range(50)])

        result = optimize_diversity_parameter(
            weights=mw,
            cov_matrices=covs,
            p_range=(0.1, 0.9),
            n_grid=5,
            refine=True,
        )
        assert result.optimal_param > 0
        assert result.method in ("grid", "grid+brent")
