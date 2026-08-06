"""Comprehensive sklearn pipeline compatibility tests.

Validates that:
- SPTTransformer fits on price DataFrame and transforms to weights
- DiversityFeature produces correct diversity measure values
- ExcessGrowthFeature matches core.excess_growth_rate on same data
- All sklearn transformers work in a Pipeline (compose with StandardScaler, PCA)
- fit → transform → verify output shape and values
- wrap_sklearn_estimator wraps GaussianProcessRegressor correctly
- GPU: SPTTransformer and DiversityFeature work with PyTorch CUDA tensors
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("sklearn")
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from quantspt.core.generating_functions import GeneratingFunction  # noqa: E402
from quantspt.integrations.sklearn import (  # noqa: E402
    DiversityFeature,
    ExcessGrowthFeature,
    SPTTransformer,
)
from quantspt.ml._protocols import GeneratingFunctionModel  # noqa: E402
from quantspt.ml.wrappers import wrap_sklearn_estimator  # noqa: E402

try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.fixture
def price_df(rng):
    """100-day price matrix for 5 assets."""
    import pandas as pd

    prices = rng.uniform(50, 150, (100, 5))
    return pd.DataFrame(prices, columns=["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"])


@pytest.fixture
def weight_matrix(rng):
    """50 observations of 5-asset market weights."""
    alpha = rng.exponential(size=(50, 5))
    return (alpha / alpha.sum(axis=1, keepdims=True)).astype(np.float64)


# ---------------------------------------------------------------------------
# Pipeline compatibility
# ---------------------------------------------------------------------------


class TestSklearnPipeline:
    """SPT transformers work inside sklearn Pipeline."""

    def test_spt_transformer_in_pipeline(self, price_df) -> None:
        """SPTTransformer composes with StandardScaler in a Pipeline."""
        pipe = Pipeline(
            [
                ("weights", SPTTransformer()),
                ("scaler", StandardScaler()),
            ]
        )
        result = pipe.fit_transform(price_df.values)
        assert result.shape == (100, 5)
        assert np.all(np.isfinite(result))

    def test_diversity_feature_in_pipeline(self, weight_matrix) -> None:
        """DiversityFeature composes with StandardScaler."""
        pipe = Pipeline(
            [
                ("diversity", DiversityFeature(p=0.5, from_weights=True)),
                ("scaler", StandardScaler()),
            ]
        )
        result = pipe.fit_transform(weight_matrix)
        assert result.shape == (50, 1)
        assert np.all(np.isfinite(result))
        assert abs(result.mean()) < 1e-10  # StandardScaler centers to 0

    def test_spt_transformer_then_pca(self, price_df) -> None:
        """SPTTransformer → PCA pipeline."""
        pipe = Pipeline(
            [
                ("weights", SPTTransformer()),
                ("pca", PCA(n_components=3)),
            ]
        )
        result = pipe.fit_transform(price_df.values)
        assert result.shape == (100, 3)
        assert np.all(np.isfinite(result))

    def test_full_feature_pipeline(self, rng) -> None:
        """SPTTransformer → DiversityFeature in sequence."""
        prices = rng.uniform(50, 200, (100, 5))
        pipe = Pipeline(
            [
                ("weights", SPTTransformer()),
                ("diversity", DiversityFeature(p=0.5, from_weights=True)),
            ]
        )
        result = pipe.fit_transform(prices)
        assert result.shape == (100, 1)
        assert np.all(result >= 1.0 - 1e-10)

    def test_pipeline_get_params(self) -> None:
        """Pipeline params are accessible for grid search."""
        pipe = Pipeline(
            [
                ("weights", SPTTransformer(normalize=True)),
                ("diversity", DiversityFeature(p=0.5)),
            ]
        )
        params = pipe.get_params()
        assert "weights__normalize" in params
        assert "diversity__p" in params

    def test_pipeline_set_params(self) -> None:
        """Pipeline params can be set (for GridSearchCV)."""
        pipe = Pipeline(
            [
                ("weights", SPTTransformer()),
                ("diversity", DiversityFeature(p=0.5)),
            ]
        )
        pipe.set_params(diversity__p=0.7)
        assert pipe.named_steps["diversity"].p == 0.7


# ---------------------------------------------------------------------------
# DiversityFeature correctness
# ---------------------------------------------------------------------------


class TestDiversityFeatureCorrectness:
    """DiversityFeature values match manual computation."""

    def test_matches_manual_diversity(self, weight_matrix) -> None:
        """DiversityFeature(p=0.5) == (Σ μ_i^0.5)^(1/0.5) = (Σ μ_i^0.5)^2."""
        feat = DiversityFeature(p=0.5, from_weights=True)
        result = feat.fit_transform(weight_matrix)
        for t in range(len(weight_matrix)):
            mu = weight_matrix[t]
            expected = np.sum(mu**0.5) ** (1.0 / 0.5)
            assert_allclose(result[t, 0], expected, rtol=1e-10)

    def test_monotone_in_p(self, weight_matrix) -> None:
        """Higher p → lower diversity (closer to 1)."""
        feat_low = DiversityFeature(p=0.3, from_weights=True)
        feat_high = DiversityFeature(p=0.8, from_weights=True)
        result_low = feat_low.fit_transform(weight_matrix)
        result_high = feat_high.fit_transform(weight_matrix)
        assert np.all(result_low >= result_high - 1e-10)


# ---------------------------------------------------------------------------
# ExcessGrowthFeature correctness
# ---------------------------------------------------------------------------


class TestExcessGrowthFeatureCorrectness:
    """ExcessGrowthFeature matches core.excess_growth_rate on same data."""

    def test_matches_core_egr(self, rng) -> None:
        """ExcessGrowthFeature result matches manual γ* computation."""
        from quantspt.core.growth_rates import excess_growth_rate

        n = 5
        T = 60
        prices = np.exp(np.cumsum(rng.normal(0, 0.02, (T, n)), axis=0)) * 100.0
        weights = prices / prices.sum(axis=1, keepdims=True)

        feat = ExcessGrowthFeature(window=20, min_periods=10, from_weights=True)
        result = feat.fit_transform(weights)

        for t in range(20, T):
            w_window = weights[t - 20 : t]
            log_ret = np.diff(np.log(w_window), axis=0)
            cov = np.cov(log_ret, rowvar=False, ddof=1)
            mu_t = weights[t]
            gamma_manual = excess_growth_rate(mu_t, cov)
            if np.isfinite(result[t, 0]) and np.isfinite(gamma_manual):
                assert_allclose(result[t, 0], gamma_manual, rtol=0.3)


# ---------------------------------------------------------------------------
# wrap_sklearn_estimator with GaussianProcessRegressor
# ---------------------------------------------------------------------------


class TestWrapGaussianProcess:
    """wrap_sklearn_estimator works with GaussianProcessRegressor."""

    def test_gpr_wrapper_basic(self, rng) -> None:
        """GPR can be wrapped and fitted."""
        T, n = 100, 5
        alpha = rng.exponential(size=(T, n))
        mw = (alpha / alpha.sum(axis=1, keepdims=True)).astype(np.float64)
        g_vals = np.sum(mw**0.5, axis=1)

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
        wrapper = wrap_sklearn_estimator(gpr, n_assets=n)
        wrapper.fit(mw, g_values=g_vals)

        mu = mw[0]
        val = wrapper.generating_function(mu)
        assert val > 0
        assert np.isfinite(val)

    def test_gpr_wrapper_gradient(self, rng) -> None:
        """GPR wrapper provides finite gradients."""
        T, n = 100, 5
        alpha = rng.exponential(size=(T, n))
        mw = (alpha / alpha.sum(axis=1, keepdims=True)).astype(np.float64)
        g_vals = np.sum(mw**0.5, axis=1)

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
        wrapper = wrap_sklearn_estimator(gpr, n_assets=n)
        wrapper.fit(mw, g_values=g_vals)

        grad = wrapper.log_gradient(mw[0])
        assert grad.shape == (5,)
        assert np.all(np.isfinite(grad))

    def test_gpr_wrapper_to_generating_function(self, rng) -> None:
        """GPR wrapper converts to GeneratingFunction."""
        T, n = 100, 5
        alpha = rng.exponential(size=(T, n))
        mw = (alpha / alpha.sum(axis=1, keepdims=True)).astype(np.float64)
        g_vals = np.sum(mw**0.5, axis=1)

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
        wrapper = wrap_sklearn_estimator(gpr, n_assets=n)
        wrapper.fit(mw, g_values=g_vals)

        G = wrapper.to_generating_function()
        assert isinstance(G, GeneratingFunction)

    def test_gpr_wrapper_implements_protocol(self) -> None:
        """GPR wrapper satisfies GeneratingFunctionModel."""
        gpr = GaussianProcessRegressor()
        wrapper = wrap_sklearn_estimator(gpr, n_assets=5)
        assert isinstance(wrapper, GeneratingFunctionModel)


# ---------------------------------------------------------------------------
# fit/transform output shape and values
# ---------------------------------------------------------------------------


class TestFitTransformShapes:
    """Verify output shapes and value ranges after fit_transform."""

    def test_spt_transformer_shape(self, rng) -> None:
        prices = rng.uniform(50, 200, (200, 10))
        t = SPTTransformer()
        result = t.fit_transform(prices)
        assert result.shape == (200, 10)
        assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(result >= 0)

    def test_diversity_feature_shape(self, rng) -> None:
        w = rng.dirichlet(np.ones(8), size=100)
        feat = DiversityFeature(p=0.5, from_weights=True)
        result = feat.fit_transform(w)
        assert result.shape == (100, 1)
        assert np.all(result >= 1.0 - 1e-10)

    def test_excess_growth_feature_shape(self, rng) -> None:
        prices = rng.uniform(50, 200, (100, 5))
        feat = ExcessGrowthFeature(window=20, min_periods=10)
        result = feat.fit_transform(prices)
        assert result.shape == (100, 1)


# ---------------------------------------------------------------------------
# GPU/CPU consistency for sklearn bridge (PyTorch tensors)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestSPTTransformerGPU:
    """SPTTransformer works with PyTorch GPU tensors directly."""

    def test_transform_cuda_tensor(self, rng) -> None:
        """SPTTransformer produces valid weights from CUDA tensors."""
        prices_np = rng.uniform(50, 200, (100, 5)).astype(np.float64)
        prices_cuda = torch.tensor(prices_np, device="cuda", dtype=torch.float64)

        transformer = SPTTransformer()
        result = transformer.transform(prices_cuda)

        assert result.device.type == "cuda"
        assert result.shape == (100, 5)
        sums = result.sum(dim=1)
        assert torch.allclose(sums, torch.ones(100, device="cuda", dtype=torch.float64))

    def test_gpu_matches_cpu(self, rng) -> None:
        """GPU and CPU produce identical weights."""
        prices_np = rng.uniform(50, 200, (50, 5)).astype(np.float64)
        prices_cuda = torch.tensor(prices_np, device="cuda", dtype=torch.float64)

        transformer = SPTTransformer()
        result_cpu = transformer.transform(prices_np)
        result_gpu = transformer.transform(prices_cuda).cpu().numpy()

        assert_allclose(result_cpu, result_gpu, atol=1e-12)

    def test_min_weight_filter_gpu(self) -> None:
        """min_weight filtering works on GPU."""
        prices = torch.tensor([[1000.0, 1.0, 1.0]], device="cuda", dtype=torch.float64)
        transformer = SPTTransformer(min_weight=0.01)
        result = transformer.transform(prices)

        assert result.device.type == "cuda"
        assert result[0, 1].item() == 0.0
        assert result[0, 2].item() == 0.0
        assert abs(result.sum().item() - 1.0) < 1e-10

    def test_fit_transform_cuda(self, rng) -> None:
        """fit_transform works end-to-end with CUDA tensor."""
        prices_cuda = torch.tensor(
            rng.uniform(50, 200, (80, 5)), device="cuda", dtype=torch.float64
        )
        transformer = SPTTransformer()
        result = transformer.fit_transform(prices_cuda)
        assert result.device.type == "cuda"
        assert result.shape == (80, 5)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestDiversityFeatureGPU:
    """DiversityFeature works with PyTorch GPU tensors."""

    def test_diversity_cuda_tensor(self, rng) -> None:
        """DiversityFeature produces valid output on CUDA tensors."""
        alpha = rng.exponential(size=(50, 5))
        weights_np = (alpha / alpha.sum(axis=1, keepdims=True)).astype(np.float64)
        weights_cuda = torch.tensor(weights_np, device="cuda", dtype=torch.float64)

        feat = DiversityFeature(p=0.5, from_weights=True)
        result = feat.transform(weights_cuda)

        assert result.device.type == "cuda"
        assert result.shape == (50, 1)
        assert torch.all(result >= 1.0 - 1e-10)

    def test_diversity_gpu_matches_cpu(self, rng) -> None:
        """GPU diversity matches CPU diversity."""
        alpha = rng.exponential(size=(30, 5))
        weights_np = (alpha / alpha.sum(axis=1, keepdims=True)).astype(np.float64)
        weights_cuda = torch.tensor(weights_np, device="cuda", dtype=torch.float64)

        feat = DiversityFeature(p=0.5, from_weights=True)
        result_cpu = feat.transform(weights_np)
        result_gpu = feat.transform(weights_cuda).cpu().numpy()

        assert_allclose(result_cpu, result_gpu, atol=1e-10)

    def test_diversity_from_prices_gpu(self, rng) -> None:
        """DiversityFeature(from_weights=False) works on CUDA price tensor."""
        prices_cuda = torch.tensor(
            rng.uniform(50, 200, (60, 5)), device="cuda", dtype=torch.float64
        )
        feat = DiversityFeature(p=0.5, from_weights=False)
        result = feat.transform(prices_cuda)

        assert result.device.type == "cuda"
        assert result.shape == (60, 1)
        assert torch.all(torch.isfinite(result))
