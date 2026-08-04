"""Statistical estimation from observed data.

Handles the *inference* side of SPT: estimating covariance rates, growth
rates, diversity parameters, and model parameters from market data.

Submodules
----------
covariance
    Suite of covariance estimators (sample, shrinkage, factor, RMT).
growth_rates
    Growth rate estimation with bias correction.
rank_statistics
    Local time estimation and transition rates.
diversity
    Diversity parameter δ estimation with confidence intervals.
model_selection
    AIC/BIC/cross-validation for model choice.
calibration
    Fit abstract models (Atlas, etc.) to data.
"""

__all__: list[str] = []
