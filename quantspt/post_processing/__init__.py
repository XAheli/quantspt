"""Post-processing utilities for portfolio optimisation results.

Submodules
----------
clean_weights
    Zero out negligible allocations, round, and enforce bounds.
discrete_allocation
    Convert continuous weights to integer share counts.
lot_sizing
    Round to exchange lot sizes, filter small trades.
export
    Serialise results to CSV, JSON, or DataFrame.
"""

from .clean_weights import clean_weights, enforce_bounds, round_weights
from .discrete_allocation import AllocationResult, greedy_allocation, lp_allocation
from .export import to_csv, to_dataframe, to_json
from .lot_sizing import minimum_trade_filter, round_to_lots

__all__ = [
    "AllocationResult",
    "clean_weights",
    "enforce_bounds",
    "greedy_allocation",
    "lp_allocation",
    "minimum_trade_filter",
    "round_to_lots",
    "round_weights",
    "to_csv",
    "to_dataframe",
    "to_json",
]
