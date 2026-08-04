"""Pure mathematical definitions for Stochastic Portfolio Theory.

This module implements exact formulas assuming *known* parameters. It never
touches data, never estimates anything, and has no side effects. Every
function is a pure mathematical mapping.

Submodules
----------
covariance
    Covariance rate process and relative covariance τ^π.
growth_rates
    Excess growth rate γ*_π and related quantities.
diversity
    Diversity measures, conditions, and coherence checks.
generating_functions
    Functionally Generated Portfolio (FGP) framework.
master_formula
    Master formula verification and decomposition.
market
    Market weight dynamics, ranked weights, and coherence.
portfolio
    Portfolio value process, relative returns, and turnover.
processes
    Stochastic process implementations and discretisation schemes.
"""

from .covariance import (
    non_degeneracy_bounds,
    portfolio_covariance_vector,
    portfolio_variance,
    relative_covariance,
    tau_bounds,
    tau_diagonal,
    verify_non_degeneracy,
)
from .diversity import (
    arbitrage_horizon_bound,
    concentration_ratio,
    diversity_deficit,
    entropy,
    herfindahl_hirschman_index,
    is_diverse,
    is_weakly_diverse,
    p_diversity,
)
from .generating_functions import (
    CustomGenerator,
    DiversityGenerator,
    EntropyGenerator,
    GeneratingFunction,
    InverseVolatilityGenerator,
    ModifiedEntropyGenerator,
    drift_process,
    fernholz_weights,
)
from .growth_rates import (
    atlas_excess_growth_rate_equal_weighted,
    atlas_excess_growth_rate_uncorrelated,
    atlas_market_growth_rate,
    excess_growth_rate,
    excess_growth_rate_bounds,
    excess_growth_rate_from_tau,
    portfolio_growth_rate,
    relative_performance_rate,
)
from .market import (
    capital_distribution_curve,
    coherence_residual,
    log_log_capital_curve,
    market_excess_growth_rate,
    market_weight_diffusion,
    market_weight_drift,
    rank_permutation,
    ranked_weights,
    validate_weights,
    verify_coherence,
)
from .master_formula import (
    boundary_term,
    drift_integral,
    master_formula_decomposition,
    verify_master_formula,
)
from .portfolio import (
    cumulative_turnover,
    drift_of_relative_return,
    holding_drift,
    log_relative_return,
    portfolio_log_return,
    portfolio_value_weights,
    rebalancing_turnover,
    relative_return,
)
from .processes import (
    CorrelatedGBM,
    EulerMaruyamaDiscretization,
    ExactGBMDiscretization,
    MilsteinDiscretization,
    simulate_path,
)

__all__ = [
    "CorrelatedGBM",
    "CustomGenerator",
    "DiversityGenerator",
    "EntropyGenerator",
    # processes
    "EulerMaruyamaDiscretization",
    "ExactGBMDiscretization",
    # generating_functions
    "GeneratingFunction",
    "InverseVolatilityGenerator",
    "MilsteinDiscretization",
    "ModifiedEntropyGenerator",
    "arbitrage_horizon_bound",
    "atlas_excess_growth_rate_equal_weighted",
    "atlas_excess_growth_rate_uncorrelated",
    "atlas_market_growth_rate",
    # master_formula
    "boundary_term",
    "capital_distribution_curve",
    "coherence_residual",
    "concentration_ratio",
    "cumulative_turnover",
    "diversity_deficit",
    "drift_integral",
    "drift_of_relative_return",
    "drift_process",
    "entropy",
    # growth_rates
    "excess_growth_rate",
    "excess_growth_rate_bounds",
    "excess_growth_rate_from_tau",
    "fernholz_weights",
    "herfindahl_hirschman_index",
    "holding_drift",
    "is_diverse",
    "is_weakly_diverse",
    "log_log_capital_curve",
    "log_relative_return",
    "market_excess_growth_rate",
    "market_weight_diffusion",
    "market_weight_drift",
    "master_formula_decomposition",
    "non_degeneracy_bounds",
    # diversity
    "p_diversity",
    "portfolio_covariance_vector",
    "portfolio_growth_rate",
    # portfolio
    "portfolio_log_return",
    "portfolio_value_weights",
    "portfolio_variance",
    "rank_permutation",
    "ranked_weights",
    "rebalancing_turnover",
    # covariance
    "relative_covariance",
    "relative_performance_rate",
    "relative_return",
    "simulate_path",
    "tau_bounds",
    "tau_diagonal",
    # market
    "validate_weights",
    "verify_coherence",
    "verify_master_formula",
    "verify_non_degeneracy",
]
