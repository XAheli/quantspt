"""Relative arbitrage theory.

Implements diversity-based arbitrage detection, mirror portfolios,
minimum horizon computation, and explicit arbitrage construction
from FKK (2005) and the Lukacs Lectures (2006).

Submodules
----------
conditions
    Diversity / weak diversity / asymptotic conditions.
detection
    Arbitrage opportunity screening.
horizon
    Minimum horizon computation (FKK Eq. 4.5).
mirror
    Mirror portfolios (FKK §8).
construction
    Explicit arbitrage portfolio construction.
"""

from quantspt.arbitrage.conditions import (
    check_asymptotic_weak_diversity,
    check_strict_diversity,
    check_weak_diversity,
    estimate_diversity_parameters,
)
from quantspt.arbitrage.construction import (
    construct_arbitrage_portfolio,
    diversity_arbitrage_portfolio,
    modified_entropy_arbitrage_portfolio,
)
from quantspt.arbitrage.detection import (
    ArbitrageOpportunity,
    check_sufficient_intrinsic_volatility,
    detect_diversity_arbitrage,
    estimate_nondegeneracy,
)
from quantspt.arbitrage.horizon import (
    diversity_horizon,
    entropy_horizon,
    horizon_sensitivity,
)
from quantspt.arbitrage.mirror import (
    mirror_covariance_rate,
    mirror_is_long_only,
    mirror_performance_residual,
    mirror_portfolio,
)

__all__ = [
    "ArbitrageOpportunity",
    "check_asymptotic_weak_diversity",
    "check_strict_diversity",
    "check_sufficient_intrinsic_volatility",
    "check_weak_diversity",
    "construct_arbitrage_portfolio",
    "detect_diversity_arbitrage",
    "diversity_arbitrage_portfolio",
    "diversity_horizon",
    "entropy_horizon",
    "estimate_diversity_parameters",
    "estimate_nondegeneracy",
    "horizon_sensitivity",
    "mirror_covariance_rate",
    "mirror_is_long_only",
    "mirror_performance_residual",
    "mirror_portfolio",
    "modified_entropy_arbitrage_portfolio",
]
