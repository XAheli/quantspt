"""Machine Learning extensions for Stochastic Portfolio Theory.

This package provides ML-powered implementations of SPT constructs:

- **Neural FGP**: Learn generating functions via Input Convex Neural Networks
  (arXiv:2506.19715, Monoyios & Pricilia 2025)
- **Wrappers**: Adapt any model (PyTorch, sklearn, callable) to SPT
- **Losses**: Composable loss functions for FGP training
- **Regime Detection**: HMM and changepoint-based market regime identification
- **Covariance Estimation**: Factor models and RMT denoising

All modules are optional — install with ``pip install quantspt[ml]``.

Quick Start
-----------
>>> from quantspt.ml import NeuralFGP, wrap_torch_model, wrap_callable
>>>
>>> # Option A: Use our built-in ICNN
>>> model = NeuralFGP(n_assets=5)
>>> model.fit(market_weights, returns=returns)
>>> G = model.to_generating_function()
>>>
>>> # Option B: Wrap any PyTorch model
>>> gf = wrap_torch_model(my_custom_nn, n_assets=5)
>>>
>>> # Option C: Wrap a plain function
>>> gf = wrap_callable(lambda mu: sum(mu**0.5))
"""

from __future__ import annotations

from ._protocols import (
    CovarianceEstimator,
    GeneratingFunctionModel,
    LearnedGeneratingFunction,
    RegimeDetector,
)
from .wrappers import (
    CallableWrapper,
    JaxFunctionWrapper,
    SklearnWrapper,
    TorchModelWrapper,
    wrap_callable,
    wrap_jax_function,
    wrap_sklearn_estimator,
    wrap_torch_model,
)

__all__ = [
    # Conditional FGP (lazy)
    "BoundaryRobustnessRegularizer",
    # Wrappers
    "CallableWrapper",
    "ConditionalFGPConfig",
    # Protocols
    "CovarianceConditionalFGP",
    "CovarianceEstimator",
    "CovarianceFeatureExtractor",
    "GeneratingFunctionModel",
    "JaxFunctionWrapper",
    "LearnedGeneratingFunction",
    "RegimeDetector",
    "SklearnWrapper",
    "TorchModelWrapper",
    "cost_optimal_p",
    "optimal_p_for_cost_level",
    "wrap_callable",
    "wrap_jax_function",
    "wrap_sklearn_estimator",
    "wrap_torch_model",
]


def __getattr__(name: str) -> object:
    """Lazy-load heavy submodules to avoid importing torch at package level."""
    if name == "NeuralFGP":
        from .neural_fgp import NeuralFGP

        return NeuralFGP
    if name == "NeuralFGPConfig":
        from .neural_fgp import NeuralFGPConfig

        return NeuralFGPConfig
    if name == "InputConvexNN":
        from .neural_fgp import InputConvexNN

        return InputConvexNN
    if name == "AdaptiveFGP":
        from .adaptive_fgp import AdaptiveFGP

        return AdaptiveFGP
    if name == "AdaptiveFGPConfig":
        from .adaptive_fgp import AdaptiveFGPConfig

        return AdaptiveFGPConfig
    if name == "CovarianceConditionalFGP":
        from .conditional_fgp import CovarianceConditionalFGP

        return CovarianceConditionalFGP
    if name == "ConditionalFGPConfig":
        from .conditional_fgp import ConditionalFGPConfig

        return ConditionalFGPConfig
    if name == "CovarianceFeatureExtractor":
        from .conditional_fgp import CovarianceFeatureExtractor

        return CovarianceFeatureExtractor
    if name == "BoundaryRobustnessRegularizer":
        from .conditional_fgp import BoundaryRobustnessRegularizer

        return BoundaryRobustnessRegularizer
    if name == "cost_optimal_p":
        from .conditional_fgp import cost_optimal_p

        return cost_optimal_p
    if name == "optimal_p_for_cost_level":
        from .conditional_fgp import optimal_p_for_cost_level

        return optimal_p_for_cost_level
    if name == "HMMRegimeDetector":
        from .regime import HMMRegimeDetector

        return HMMRegimeDetector
    if name == "ChangepointDetector":
        from .regime import ChangepointDetector

        return ChangepointDetector
    if name == "FactorModelEstimator":
        from .covariance import FactorModelEstimator

        return FactorModelEstimator
    if name == "RMTDenoiser":
        from .covariance import RMTDenoiser

        return RMTDenoiser
    # Losses
    if name in (
        "relative_return_loss",
        "weight_regularization",
        "turnover_penalty",
        "sharpe_of_relative_loss",
        "default_loss",
        "drift_integral_loss",
        "DriftIntegralLoss",
    ):
        from . import losses

        return getattr(losses, name)
    raise AttributeError(f"module 'quantspt.ml' has no attribute {name!r}")
