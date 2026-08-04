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
ergodic
    Ergodic properties of rank-based models (BFK Prop. 2.3).
rank_portfolios
    Rank-based portfolio construction.
transitions
    Rank transition matrices and rates.
"""

__all__: list[str] = []
