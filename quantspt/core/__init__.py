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
from .master_formula import (
    boundary_term,
    drift_integral,
    master_formula_decomposition,
    verify_master_formula,
)

__all__ = [
    # covariance
    "relative_covariance",
    "portfolio_variance",
    "portfolio_covariance_vector",
    "non_degeneracy_bounds",
    "verify_non_degeneracy",
    "tau_diagonal",
    "tau_bounds",
    # growth_rates
    "excess_growth_rate",
    "excess_growth_rate_from_tau",
    "portfolio_growth_rate",
    "relative_performance_rate",
    "excess_growth_rate_bounds",
    "atlas_excess_growth_rate_uncorrelated",
    "atlas_excess_growth_rate_equal_weighted",
    "atlas_market_growth_rate",
    # diversity
    "p_diversity",
    "entropy",
    "herfindahl_hirschman_index",
    "concentration_ratio",
    "is_diverse",
    "is_weakly_diverse",
    "diversity_deficit",
    "arbitrage_horizon_bound",
    # generating_functions
    "GeneratingFunction",
    "DiversityGenerator",
    "EntropyGenerator",
    "ModifiedEntropyGenerator",
    "InverseVolatilityGenerator",
    "CustomGenerator",
    "fernholz_weights",
    "drift_process",
    # master_formula
    "boundary_term",
    "drift_integral",
    "master_formula_decomposition",
    "verify_master_formula",
]
