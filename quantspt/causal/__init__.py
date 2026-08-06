"""Causal inference extensions for Stochastic Portfolio Theory.

This package provides causal-structure-aware tools for:

- **Structure Learning**: Discover causal DAGs from observational data
  using constraint-based (PC) or score-based (GES, HillClimb) algorithms
- **Causal Covariance**: Estimate observational and interventional
  covariance matrices via linear Gaussian Bayesian networks
- **Causal Factor Models**: Extract causal factor loadings (B matrix)
  and noise covariance (Ω) from DAG structure
- **Causal Rank Dynamics**: Granger-style causal analysis on stock rank
  time series

All modules are optional — install with ``pip install quantspt[causal]``.

Quick Start
-----------
>>> from quantspt.causal import CausalStructureLearner, CausalCovarianceEstimator
>>>
>>> # Learn causal structure
>>> learner = CausalStructureLearner(method="pc", ci_test="pearsonr")
>>> learner.fit(returns_df)
>>>
>>> # Estimate causal covariance
>>> estimator = CausalCovarianceEstimator(edges=learner.edges)
>>> estimator.fit(returns_df)
>>> sigma = estimator.observational_covariance()
"""

from __future__ import annotations

from ._protocols import (
    CausalCovarianceModel,
    CausalFactorModelProtocol,
    CausalStructureModel,
)
from .covariance import CausalCovarianceEstimator, CovarianceDecomposition
from .factors import CausalFactorModel
from .rank import CausalRankAnalysis
from .structure import CausalStructureLearner

__all__ = [
    "CausalCovarianceEstimator",
    "CausalCovarianceModel",
    "CausalFactorModel",
    "CausalFactorModelProtocol",
    "CausalRankAnalysis",
    "CausalStructureLearner",
    "CausalStructureModel",
    "CovarianceDecomposition",
]
