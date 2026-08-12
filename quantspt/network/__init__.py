"""Financial network construction, centrality analysis, and contagion modeling.

Spatial pillar of the spatial-temporal microstructure expansion. Builds
weighted directed graphs from return data and propagates shocks through
balance-sheet or exposure-based linkages.
"""

from __future__ import annotations

from quantspt.network.contagion import (
    ContagionResult,
    DebtRank,
    EisenbergNoe,
    clearing_vector,
    debt_rank,
)
from quantspt.network.topology import (
    FinancialNetwork,
    NetworkMetrics,
    build_granger_network,
    build_partial_correlation_network,
    build_transfer_entropy_network,
    compute_centrality,
)

__all__ = [
    "ContagionResult",
    "DebtRank",
    "EisenbergNoe",
    "FinancialNetwork",
    "NetworkMetrics",
    "build_granger_network",
    "build_partial_correlation_network",
    "build_transfer_entropy_network",
    "clearing_vector",
    "compute_centrality",
    "debt_rank",
]
