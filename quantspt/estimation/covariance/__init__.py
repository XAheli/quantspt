"""Covariance rate estimation suite.

Submodules
----------
sample
    Sample covariance with annualisation and rolling windows.
shrinkage
    Ledoit-Wolf and Oracle Approximating Shrinkage estimators.
"""

from .sample import rolling_sample_covariance, sample_covariance
from .shrinkage import ledoit_wolf, oracle_approximating_shrinkage

__all__ = [
    "ledoit_wolf",
    "oracle_approximating_shrinkage",
    "rolling_sample_covariance",
    "sample_covariance",
]
