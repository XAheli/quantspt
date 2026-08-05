"""Statistical estimation from observed data.

Handles the *inference* side of SPT: estimating covariance rates, growth
rates, diversity parameters, and model parameters from market data.

Submodules
----------
covariance
    Suite of covariance estimators (sample, shrinkage).
growth_rates
    Growth rate estimation with bias correction.
diversity
    Diversity parameter δ estimation with confidence intervals.
calibration
    Fit abstract models (Atlas, etc.) to data.
"""

from .calibration import calibrate_atlas, goodness_of_fit
from .covariance import (
    ledoit_wolf,
    oracle_approximating_shrinkage,
    rolling_sample_covariance,
    sample_covariance,
)
from .diversity import (
    bootstrap_diversity_ci,
    estimate_diversity_parameter,
    rolling_diversity_deficit,
)
from .growth_rates import estimate_growth_rates, rolling_growth_rates

__all__ = [
    "bootstrap_diversity_ci",
    "calibrate_atlas",
    "estimate_diversity_parameter",
    "estimate_growth_rates",
    "goodness_of_fit",
    "ledoit_wolf",
    "oracle_approximating_shrinkage",
    "rolling_diversity_deficit",
    "rolling_growth_rates",
    "rolling_sample_covariance",
    "sample_covariance",
]
