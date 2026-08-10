"""Experimental / research-stage SPT strategies.

This package contains strategies that are theoretically interesting but
have NOT been demonstrated to outperform simpler approaches on real data.

Research finding (Aug 2026, 50 S&P 500 stocks, 2020-2026):
  Neural generating functions are information-theoretically limited on
  available data.  The optimal p sequence has lag-1 autocorrelation of
  ~0.009 (essentially white noise), requiring ~100+ years to distinguish
  adjacent p values at 80% power.

**For production use, prefer ``DiversityGenerator(p=0.3)`` with
universe selection via ``quantspt.universe.SPTUniverseSelector``.**

Modules
-------
- ``neural_fgp`` — ICNN-based learned generating functions
- ``adaptive_fgp`` — correction-anchored adaptive generating functions
- ``conditional_fgp`` — covariance-conditional generating functions
"""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """Lazy-load heavy experimental modules to avoid importing torch."""
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
    raise AttributeError(f"module 'quantspt.experimental' has no attribute {name!r}")


__all__ = [
    "AdaptiveFGP",
    "AdaptiveFGPConfig",
    "BoundaryRobustnessRegularizer",
    "ConditionalFGPConfig",
    "CovarianceConditionalFGP",
    "CovarianceFeatureExtractor",
    "InputConvexNN",
    "NeuralFGP",
    "NeuralFGPConfig",
    "cost_optimal_p",
    "optimal_p_for_cost_level",
]
