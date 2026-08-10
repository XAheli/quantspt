"""Machine Learning extensions for Stochastic Portfolio Theory.

Production-ready modules
------------------------
- **Wrappers**: Adapt any model (PyTorch, sklearn, callable) to SPT
- **Losses**: Composable loss functions for FGP training
- **Regime Detection**: HMM and changepoint-based market regime identification
- **Covariance Estimation**: Factor models and RMT denoising

Experimental / research modules (moved to ``quantspt.experimental``)
---------------------------------------------------------------------
- **NeuralFGP**, **AdaptiveFGP**, **CovarianceConditionalFGP**

Neural generating functions are a research direction.  Empirical
investigation on 50 S&P 500 stocks (2020–2026) showed that the optimal
generating-function parameter is unpredictable (lag-1 autocorrelation ~0)
and requires ~100+ years of data to distinguish adjacent values at 80%
power.  **For production use, prefer ``DiversityGenerator`` with universe
selection via ``quantspt.universe.SPTUniverseSelector``.**

These classes are still importable from ``quantspt.ml`` for backward
compatibility, but canonical imports are from ``quantspt.experimental``.

All modules are optional — install with ``pip install quantspt[ml]``.
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
    """Lazy-load heavy submodules to avoid importing torch at package level.

    Neural/adaptive/conditional FGP classes have moved to
    ``quantspt.experimental`` but remain importable here for backward
    compatibility.
    """
    # --- Experimental (backward-compat re-exports) ---
    if name == "NeuralFGP":
        from ..experimental.neural_fgp import NeuralFGP

        return NeuralFGP
    if name == "NeuralFGPConfig":
        from ..experimental.neural_fgp import NeuralFGPConfig

        return NeuralFGPConfig
    if name == "InputConvexNN":
        from ..experimental.neural_fgp import InputConvexNN

        return InputConvexNN
    if name == "AdaptiveFGP":
        from ..experimental.adaptive_fgp import AdaptiveFGP

        return AdaptiveFGP
    if name == "AdaptiveFGPConfig":
        from ..experimental.adaptive_fgp import AdaptiveFGPConfig

        return AdaptiveFGPConfig
    if name == "CovarianceConditionalFGP":
        from ..experimental.conditional_fgp import CovarianceConditionalFGP

        return CovarianceConditionalFGP
    if name == "ConditionalFGPConfig":
        from ..experimental.conditional_fgp import ConditionalFGPConfig

        return ConditionalFGPConfig
    if name == "CovarianceFeatureExtractor":
        from ..experimental.conditional_fgp import CovarianceFeatureExtractor

        return CovarianceFeatureExtractor
    if name == "BoundaryRobustnessRegularizer":
        from ..experimental.conditional_fgp import BoundaryRobustnessRegularizer

        return BoundaryRobustnessRegularizer
    if name == "cost_optimal_p":
        from ..experimental.conditional_fgp import cost_optimal_p

        return cost_optimal_p
    if name == "optimal_p_for_cost_level":
        from ..experimental.conditional_fgp import optimal_p_for_cost_level

        return optimal_p_for_cost_level
    # --- Production modules ---
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
