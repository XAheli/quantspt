"""Rank-based theory and dynamics.

Implements the ranked capitalisation processes, local time estimation,
capital distribution analysis, and rank-based portfolio construction
from BFK (2005) and F&K Survey (2008).

Submodules
----------
processes
    Ranked capitalisation processes Z_k(t) (BFK §3).
local_times
    Local time estimation Λ_{k,k+1}(t) (BFK Eq. 3.3).
capital_distribution
    Pareto fits and stability analysis (BFK §4).
rank_portfolios
    Rank-based portfolio construction.
transitions
    Rank transition matrices and rates.
"""

from quantspt.rank.capital_distribution import (
    capital_distribution_curve,
    capital_distribution_stability,
    log_log_fit,
    pareto_exponents_empirical,
)
from quantspt.rank.local_times import (
    empirical_local_time_rates,
    empirical_local_times,
    local_time_rates_analytical,
    local_time_rates_atlas,
)
from quantspt.rank.processes import (
    rank_assignment,
    ranked_capitalizations,
    ranked_capitalizations_path,
    ranked_drift_coefficients,
    ranked_weights_from_path,
)
from quantspt.rank.rank_portfolios import (
    bottom_m_portfolio,
    leaking_portfolio,
    rank_weighted_portfolio,
    top_m_portfolio,
)
from quantspt.rank.transitions import (
    expected_sojourn_times,
    rank_mobility_index,
    rank_transition_matrix,
)

__all__ = [
    "bottom_m_portfolio",
    "capital_distribution_curve",
    "capital_distribution_stability",
    "empirical_local_time_rates",
    "empirical_local_times",
    "expected_sojourn_times",
    "leaking_portfolio",
    "local_time_rates_analytical",
    "local_time_rates_atlas",
    "log_log_fit",
    "pareto_exponents_empirical",
    "rank_assignment",
    "rank_mobility_index",
    "rank_transition_matrix",
    "rank_weighted_portfolio",
    "ranked_capitalizations",
    "ranked_capitalizations_path",
    "ranked_drift_coefficients",
    "ranked_weights_from_path",
    "top_m_portfolio",
]
