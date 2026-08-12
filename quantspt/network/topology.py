"""Financial network construction and centrality analysis.

Builds weighted directed graphs from asset return data using three
complementary methods that capture different facets of financial linkage:
partial correlation (contemporaneous conditional dependence), Granger
causality (predictive lead-lag), and transfer entropy (non-linear
information flow).

Mathematical References
-----------------------
- Graphical LASSO (GLASSO): Friedman, Hastie & Tibshirani (2008),
  "Sparse inverse covariance estimation with the graphical lasso,"
  Biostatistics 9(3), pp. 432-441.
  Solves  min_Θ {-log det Θ + tr(SΘ) + λ‖Θ‖₁}  where Θ = Σ⁻¹ is the
  precision matrix; off-diagonal entries of Θ give partial correlations.

- Granger causality: Granger (1969), "Investigating Causal Relations by
  Econometric Models and Cross-spectral Methods," Econometrica 37(3).
  Tests whether past values of Y improve prediction of X beyond its own
  lags via F-test on restricted vs. unrestricted VAR residuals.

- Transfer entropy: Schreiber (2000), "Measuring Information Transfer,"
  Phys. Rev. Lett. 85, pp. 461-464.  TE(Y→X) = H(X_t | X_{t-1:t-k})
  - H(X_t | X_{t-1:t-k}, Y_{t-1:t-k}); estimated via k-nearest-neighbor
  entropy estimator (Kraskov et al., 2004).

- Centrality: Freeman (1977) for betweenness; Bonacich (1972) for
  eigenvector; Brin & Page (1998) for PageRank.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._preconditions import require
from .._result import SPTResult

__all__ = [
    "FinancialNetwork",
    "NetworkMetrics",
    "build_granger_network",
    "build_partial_correlation_network",
    "build_transfer_entropy_network",
    "compute_centrality",
]


@dataclass(frozen=True)
class NetworkMetrics:
    """Centrality and topological metrics for each node in a financial network.

    Attributes
    ----------
    degree : dict[str, float]
        Weighted degree centrality (normalized by max degree).
    eigenvector : dict[str, float]
        Eigenvector centrality (influence in the dominant eigenspace).
    betweenness : dict[str, float]
        Betweenness centrality (fraction of shortest paths through node).
        Computed using ``distance = 1/weight`` so that higher-weight edges
        are treated as shorter (stronger) paths.
    pagerank : dict[str, float]
        PageRank scores (recursive importance via random walk).
    in_degree : dict[str, float] or None
        Weighted in-degree centrality for directed graphs (None for
        undirected).
    out_degree : dict[str, float] or None
        Weighted out-degree centrality for directed graphs (None for
        undirected).
    """

    degree: dict[str, float]
    eigenvector: dict[str, float]
    betweenness: dict[str, float]
    pagerank: dict[str, float]
    in_degree: dict[str, float] | None = None
    out_degree: dict[str, float] | None = None


@dataclass(frozen=True)
class FinancialNetwork:
    """Container for a financial network with associated metrics.

    Attributes
    ----------
    graph : nx.Graph or nx.DiGraph
        Weighted graph. Edge attribute ``'weight'`` encodes the strength
        of the financial linkage (partial correlation, F-statistic, or
        transfer entropy). Undirected for symmetric methods (GLASSO),
        directed for causal methods (Granger, transfer entropy).
    adjacency : NDArray[np.float64]
        Dense adjacency matrix of shape ``(n, n)`` in node-label order.
    node_names : list[str]
        Ordered node labels matching rows/columns of ``adjacency``.
    method : str
        Construction method that produced this network.
    metrics : NetworkMetrics or None
        Centrality metrics, populated when ``compute_centrality`` is called.
    """

    graph: nx.Graph | nx.DiGraph
    adjacency: NDArray[np.float64]
    node_names: list[str]
    method: str
    metrics: NetworkMetrics | None = None


# ---------------------------------------------------------------------------
# Network construction: Partial Correlation (GLASSO)
# ---------------------------------------------------------------------------


def build_partial_correlation_network(
    returns: pd.DataFrame | NDArray[np.float64],
    *,
    alpha: float = 0.01,
    node_names: list[str] | None = None,
    threshold: float = 0.0,
    max_iter: int = 500,
) -> SPTResult[FinancialNetwork]:
    r"""Build a financial network from partial correlations via GLASSO.

    Estimates the precision matrix Θ = Σ⁻¹ using the graphical LASSO
    (Friedman et al., 2008), then extracts partial correlations as:

    .. math::
        \rho_{ij|\text{rest}} = -\frac{\Theta_{ij}}
        {\sqrt{\Theta_{ii}\,\Theta_{jj}}}

    Edges are placed where |ρ_partial| exceeds *threshold*.

    Parameters
    ----------
    returns : DataFrame of shape (T, n) or ndarray
        Log-return time series. Columns are assets, rows are observations.
    alpha : float
        L1 regularization strength for GLASSO. Larger values produce
        sparser networks.
    node_names : list of str, optional
        Asset labels. Required when *returns* is an ndarray.
    threshold : float
        Minimum absolute partial correlation to retain an edge.
    max_iter : int
        Maximum GLASSO iterations.

    Returns
    -------
    SPTResult[FinancialNetwork]
        Network where edge weights are absolute partial correlations.

    References
    ----------
    Friedman, Hastie & Tibshirani (2008), "Sparse inverse covariance
    estimation with the graphical lasso," Biostatistics 9(3), Eq. (1).
    """
    t0 = time.perf_counter()
    returns_arr, names = _validate_returns(returns, node_names)
    n = returns_arr.shape[1]
    require(n >= 2, f"Need at least 2 assets, got {n}")

    S = np.cov(returns_arr, rowvar=False)
    S = (S + S.T) / 2.0

    from sklearn.covariance import graphical_lasso

    precision, _ = graphical_lasso(S, alpha=alpha, max_iter=max_iter)

    d = np.sqrt(np.diag(precision))
    d_safe = np.where(d > 1e-15, d, 1.0)
    partial_corr = -precision / np.outer(d_safe, d_safe)
    np.fill_diagonal(partial_corr, 0.0)

    G = nx.Graph()
    G.add_nodes_from(names)
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(partial_corr[i, j])
            if w > threshold:
                G.add_edge(names[i], names[j], weight=w)

    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i, j] = (
                    abs(partial_corr[i, j])
                    if abs(partial_corr[i, j]) > threshold
                    else 0.0
                )

    elapsed = (time.perf_counter() - t0) * 1000.0
    net = FinancialNetwork(
        graph=G,
        adjacency=adj,
        node_names=list(names),
        method="partial_correlation_glasso",
    )
    return SPTResult(
        data=net,
        metadata={
            "method": "GLASSO",
            "alpha": alpha,
            "threshold": threshold,
            "n_edges": G.number_of_edges(),
        },
        computation_time_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Network construction: Granger Causality
# ---------------------------------------------------------------------------


def build_granger_network(
    returns: pd.DataFrame | NDArray[np.float64],
    *,
    max_lag: int = 5,
    significance: float = 0.05,
    node_names: list[str] | None = None,
) -> SPTResult[FinancialNetwork]:
    r"""Build a financial network from pairwise Granger causality tests.

    For each ordered pair (Y, X), tests whether past values of Y improve
    the prediction of X via an F-test comparing unrestricted and restricted
    VAR residual sums of squares (Granger, 1969).

    A directed edge Y → X is placed when the null hypothesis of no Granger
    causality is rejected at the given significance level. The edge weight
    is −log₁₀(p-value), so stronger causality produces higher weights.

    Parameters
    ----------
    returns : DataFrame of shape (T, n) or ndarray
        Return time series.
    max_lag : int
        Maximum lag order for the VAR models.
    significance : float
        P-value threshold for edge inclusion.
    node_names : list of str, optional
        Asset labels when *returns* is an ndarray.

    Returns
    -------
    SPTResult[FinancialNetwork]
        Directed network with −log₁₀(p) edge weights.

    References
    ----------
    Granger (1969), "Investigating Causal Relations by Econometric Models
    and Cross-spectral Methods," Econometrica 37(3), pp. 424-438.
    """
    t0 = time.perf_counter()
    returns_arr, names = _validate_returns(returns, node_names)
    n = returns_arr.shape[1]
    T = returns_arr.shape[0]
    require(n >= 2, f"Need at least 2 assets, got {n}")
    require(
        2 * max_lag + 2 < T, f"Need at least {2 * max_lag + 3} observations, got {T}"
    )

    import warnings as _warnings

    from statsmodels.tsa.stattools import grangercausalitytests

    G = nx.DiGraph()
    G.add_nodes_from(names)
    adj = np.zeros((n, n), dtype=np.float64)
    warn_list: list[str] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair_data = np.column_stack([returns_arr[:, i], returns_arr[:, j]])
            try:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", FutureWarning)
                    result = grangercausalitytests(
                        pair_data, maxlag=max_lag, verbose=False
                    )
                best_p = min(
                    result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)
                )
                if best_p < significance:
                    weight = -np.log10(max(best_p, 1e-300))
                    G.add_edge(names[j], names[i], weight=weight)
                    adj[j, i] = weight
            except Exception as e:
                warn_list.append(f"Granger test {names[j]}→{names[i]} failed: {e}")

    elapsed = (time.perf_counter() - t0) * 1000.0
    net = FinancialNetwork(
        graph=G, adjacency=adj, node_names=list(names), method="granger_causality"
    )
    return SPTResult(
        data=net,
        metadata={
            "method": "Granger",
            "max_lag": max_lag,
            "significance": significance,
            "n_edges": G.number_of_edges(),
        },
        warnings=warn_list,
        computation_time_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Network construction: Transfer Entropy
# ---------------------------------------------------------------------------


def build_transfer_entropy_network(
    returns: pd.DataFrame | NDArray[np.float64],
    *,
    lag: int = 1,
    n_bins: int = 6,
    node_names: list[str] | None = None,
    threshold: float = 0.0,
) -> SPTResult[FinancialNetwork]:
    r"""Build a financial network from pairwise transfer entropy estimates.

    Transfer entropy from Y to X at lag k is defined as (Schreiber, 2000):

    .. math::
        TE(Y \to X) = H(X_t \mid X_{t-1:t-k}) - H(X_t \mid X_{t-1:t-k},\, Y_{t-1:t-k})

    Estimated via plug-in histogram estimator with *n_bins* equiprobable
    bins (rank-based discretization).

    Parameters
    ----------
    returns : DataFrame of shape (T, n) or ndarray
        Return time series.
    lag : int
        Number of lagged values to condition on.
    n_bins : int
        Number of equiprobable bins for discretization.
    node_names : list of str, optional
        Asset labels when *returns* is an ndarray.
    threshold : float
        Minimum TE value to retain an edge.

    Returns
    -------
    SPTResult[FinancialNetwork]
        Directed network with TE edge weights.

    References
    ----------
    Schreiber (2000), "Measuring Information Transfer," Phys. Rev. Lett.
    85, pp. 461-464.
    """
    t0 = time.perf_counter()
    returns_arr, names = _validate_returns(returns, node_names)
    n = returns_arr.shape[1]
    T = returns_arr.shape[0]
    require(n >= 2, f"Need at least 2 assets, got {n}")
    require(lag + 1 < T, f"Need at least {lag + 2} observations, got {T}")

    G = nx.DiGraph()
    G.add_nodes_from(names)
    adj = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tent = _transfer_entropy(
                returns_arr[:, i], returns_arr[:, j], lag=lag, n_bins=n_bins
            )
            if tent > threshold:
                G.add_edge(names[j], names[i], weight=tent)
                adj[j, i] = tent

    elapsed = (time.perf_counter() - t0) * 1000.0
    net = FinancialNetwork(
        graph=G, adjacency=adj, node_names=list(names), method="transfer_entropy"
    )
    return SPTResult(
        data=net,
        metadata={
            "method": "TransferEntropy",
            "lag": lag,
            "n_bins": n_bins,
            "n_edges": G.number_of_edges(),
        },
        computation_time_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Centrality computation
# ---------------------------------------------------------------------------


def compute_centrality(network: FinancialNetwork) -> FinancialNetwork:
    """Compute centrality metrics for all nodes in a financial network.

    Returns a new ``FinancialNetwork`` with the ``metrics`` field populated.
    Computes degree, eigenvector, betweenness, and PageRank centrality.

    For betweenness centrality, edge weights represent connection *strength*
    (higher = stronger link), but NetworkX shortest-path algorithms treat
    the weight attribute as *cost/distance*.  We therefore compute
    ``distance = 1 / weight`` and pass ``weight="distance"`` so that
    stronger connections correspond to shorter paths.

    For directed graphs (Granger, transfer entropy), both ``in_degree`` and
    ``out_degree`` are provided separately in addition to total ``degree``.

    Parameters
    ----------
    network : FinancialNetwork
        Network to analyze.

    Returns
    -------
    FinancialNetwork
        Copy of *network* with ``metrics`` set.

    References
    ----------
    - Degree: Freeman (1977), "A Set of Measures of Centrality Based on
      Betweenness," Sociometry 40(1).
    - Eigenvector: Bonacich (1972), "Factoring and weighting approaches to
      status scores and clique identification," J. Math. Sociology 2(1).
    - PageRank: Brin & Page (1998), "The anatomy of a large-scale
      hypertextual web search engine," Computer Networks 30(1-7).
    """
    G = network.graph
    n_nodes = G.number_of_nodes()

    degree = dict(G.degree(weight="weight"))
    if n_nodes > 1:
        max_deg = max(degree.values()) if degree else 1.0
        degree = {k: v / max_deg for k, v in degree.items()}

    is_directed = G.is_directed()
    in_deg: dict[str, float] | None = None
    out_deg: dict[str, float] | None = None
    if is_directed:
        raw_in = dict(G.in_degree(weight="weight"))
        raw_out = dict(G.out_degree(weight="weight"))
        max_in = max(raw_in.values()) if raw_in and max(raw_in.values()) > 0 else 1.0
        max_out = (
            max(raw_out.values()) if raw_out and max(raw_out.values()) > 0 else 1.0
        )
        in_deg = {k: v / max_in for k, v in raw_in.items()}
        out_deg = {k: v / max_out for k, v in raw_out.items()}

    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
    except nx.NetworkXException:
        # PageRank is more robust for non-strongly-connected digraphs
        # since it adds a damping teleportation term.
        eigenvector = {node: 1.0 / n_nodes for node in G.nodes()}

    for _u, _v, data in G.edges(data=True):
        data["distance"] = 1.0 / (data["weight"] + 1e-15)
    betweenness = nx.betweenness_centrality(G, weight="distance")

    pagerank = nx.pagerank(G, weight="weight")

    metrics = NetworkMetrics(
        degree=degree,
        eigenvector=eigenvector,
        betweenness=betweenness,
        pagerank=pagerank,
        in_degree=in_deg,
        out_degree=out_deg,
    )

    return FinancialNetwork(
        graph=network.graph,
        adjacency=network.adjacency,
        node_names=network.node_names,
        method=network.method,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_returns(
    returns: pd.DataFrame | NDArray[np.float64],
    node_names: list[str] | None,
) -> tuple[NDArray[np.float64], list[str]]:
    """Convert returns to ndarray + names list with validation."""
    arr: NDArray[np.float64]
    if isinstance(returns, pd.DataFrame):
        names = list(returns.columns)
        arr = returns.values.astype(np.float64)
    else:
        arr = np.asarray(returns, dtype=np.float64)
        require(arr.ndim == 2, f"returns must be 2-D, got {arr.ndim}-D")
        if node_names is not None:
            require(
                len(node_names) == arr.shape[1],
                f"node_names length {len(node_names)} != {arr.shape[1]} columns",
            )
            names = list(node_names)
        else:
            names = [f"asset_{i}" for i in range(arr.shape[1])]

    require(arr.ndim == 2, f"returns must be 2-D, got shape {arr.shape}")
    require(
        bool(np.all(np.isfinite(arr))),
        "returns contain NaN or Inf values",
    )
    return arr, names


def _transfer_entropy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    lag: int = 1,
    n_bins: int = 6,
) -> float:
    r"""Estimate transfer entropy TE(Y→X) via histogram plug-in estimator.

    Discretizes using rank-based equiprobable binning, then computes:
    TE = H(X_t, X_{t-1:t-k}) + H(X_{t-1:t-k}, Y_{t-1:t-k})
       - H(X_{t-1:t-k}) - H(X_t, X_{t-1:t-k}, Y_{t-1:t-k})
    """
    T = len(x)
    if lag >= T:
        return 0.0

    def _digitize(arr: NDArray[np.float64]) -> NDArray[np.int64]:
        ranks = np.argsort(np.argsort(arr))
        return np.clip(ranks * n_bins // len(arr), 0, n_bins - 1).astype(np.int64)

    x_t = _digitize(x[lag:])
    x_past = np.column_stack(
        [_digitize(x[lag - k - 1 : T - k - 1]) for k in range(lag)]
    )
    y_past = np.column_stack(
        [_digitize(y[lag - k - 1 : T - k - 1]) for k in range(lag)]
    )

    def _entropy(*arrays: NDArray[np.int64]) -> float:
        combined = np.column_stack(arrays)
        _, counts = np.unique(combined, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-300)))

    h_xt_xpast = _entropy(x_t, x_past)
    h_xpast_ypast = _entropy(x_past, y_past)
    h_xpast = _entropy(x_past)
    h_xt_xpast_ypast = _entropy(x_t, x_past, y_past)

    transfer_ent = h_xt_xpast + h_xpast_ypast - h_xpast - h_xt_xpast_ypast
    return max(transfer_ent, 0.0)
