"""Tests for quantspt.network.topology — financial network construction.

Validates partial correlation, Granger causality, and transfer entropy
network builders against known structures, and verifies centrality
metric computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantspt._result import SPTResult
from quantspt.network.topology import (
    FinancialNetwork,
    NetworkMetrics,
    build_granger_network,
    build_partial_correlation_network,
    build_transfer_entropy_network,
    compute_centrality,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def block_correlated_returns(rng: np.random.Generator) -> pd.DataFrame:
    """Two clear clusters: assets 0-2 correlated, assets 3-4 correlated."""
    T = 500
    factor_a = rng.standard_normal(T)
    factor_b = rng.standard_normal(T)
    noise = rng.standard_normal((T, 5)) * 0.3

    data = np.zeros((T, 5))
    for i in range(3):
        data[:, i] = factor_a + noise[:, i]
    for i in range(3, 5):
        data[:, i] = factor_b + noise[:, i]

    return pd.DataFrame(data, columns=["A", "B", "C", "D", "E"])


@pytest.fixture()
def granger_returns() -> pd.DataFrame:
    """X Granger-causes Y (strong lagged influence)."""
    rng = np.random.default_rng(99)
    T = 2000
    x = rng.standard_normal(T)
    y = np.zeros(T)
    z = rng.standard_normal(T)
    for t in range(1, T):
        y[t] = 0.8 * x[t - 1] + rng.standard_normal() * 0.3
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


@pytest.fixture()
def simple_returns(rng: np.random.Generator) -> np.ndarray:
    """Simple 3-asset return matrix."""
    return rng.standard_normal((200, 3)) * 0.01


# ---------------------------------------------------------------------------
# Partial Correlation Network
# ---------------------------------------------------------------------------


class TestPartialCorrelationNetwork:
    def test_result_type(self, block_correlated_returns: pd.DataFrame) -> None:
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.1)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, FinancialNetwork)

    def test_network_structure(self, block_correlated_returns: pd.DataFrame) -> None:
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.05)
        net = result.data
        assert net.graph.number_of_nodes() == 5
        assert net.adjacency.shape == (5, 5)
        assert net.method == "partial_correlation_glasso"
        assert net.node_names == ["A", "B", "C", "D", "E"]

    def test_detects_cluster_structure(
        self, block_correlated_returns: pd.DataFrame
    ) -> None:
        """Within-cluster edges should be stronger than cross-cluster."""
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.01)
        adj = result.data.adjacency
        within_a = adj[0, 1] + adj[0, 2] + adj[1, 2]
        cross = adj[0, 3] + adj[0, 4] + adj[1, 3]
        assert within_a > cross, "Within-cluster connections should be stronger"

    def test_adjacency_nonnegative(
        self, block_correlated_returns: pd.DataFrame
    ) -> None:
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.1)
        assert np.all(result.data.adjacency >= 0)

    def test_diagonal_zero(self, block_correlated_returns: pd.DataFrame) -> None:
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.1)
        assert_allclose(np.diag(result.data.adjacency), 0.0)

    def test_threshold_filters_edges(
        self, block_correlated_returns: pd.DataFrame
    ) -> None:
        low = build_partial_correlation_network(
            block_correlated_returns, alpha=0.05, threshold=0.01
        )
        high = build_partial_correlation_network(
            block_correlated_returns, alpha=0.05, threshold=0.3
        )
        assert high.data.graph.number_of_edges() <= low.data.graph.number_of_edges()

    def test_ndarray_input(self, simple_returns: np.ndarray) -> None:
        result = build_partial_correlation_network(
            simple_returns, alpha=0.1, node_names=["X", "Y", "Z"]
        )
        assert result.data.node_names == ["X", "Y", "Z"]

    def test_auto_names(self, simple_returns: np.ndarray) -> None:
        result = build_partial_correlation_network(simple_returns, alpha=0.1)
        assert result.data.node_names == ["asset_0", "asset_1", "asset_2"]

    def test_metadata_populated(self, block_correlated_returns: pd.DataFrame) -> None:
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.05)
        assert "n_edges" in result.metadata
        assert result.metadata["method"] == "GLASSO"
        assert result.computation_time_ms > 0


# ---------------------------------------------------------------------------
# Granger Causality Network
# ---------------------------------------------------------------------------


class TestGrangerNetwork:
    def test_detects_causal_link(self, granger_returns: pd.DataFrame) -> None:
        result = build_granger_network(granger_returns, max_lag=3, significance=0.05)
        net = result.data
        assert net.graph.has_edge("X", "Y"), "Should detect X→Y Granger causality"

    def test_no_spurious_reverse(self, granger_returns: pd.DataFrame) -> None:
        result = build_granger_network(granger_returns, max_lag=3, significance=0.01)
        net = result.data
        assert not net.graph.has_edge("Y", "X"), "Y should not Granger-cause X"

    def test_result_type(self, granger_returns: pd.DataFrame) -> None:
        result = build_granger_network(granger_returns, max_lag=2)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, FinancialNetwork)
        assert result.data.method == "granger_causality"

    def test_edge_weights_positive(self, granger_returns: pd.DataFrame) -> None:
        result = build_granger_network(granger_returns, max_lag=2, significance=0.1)
        for _, _, data in result.data.graph.edges(data=True):
            assert data["weight"] > 0


# ---------------------------------------------------------------------------
# Transfer Entropy Network
# ---------------------------------------------------------------------------


class TestTransferEntropyNetwork:
    def test_basic_structure(self, block_correlated_returns: pd.DataFrame) -> None:
        result = build_transfer_entropy_network(
            block_correlated_returns, lag=1, n_bins=4
        )
        net = result.data
        assert net.graph.number_of_nodes() == 5
        assert net.method == "transfer_entropy"

    def test_nonnegative_weights(self, block_correlated_returns: pd.DataFrame) -> None:
        result = build_transfer_entropy_network(block_correlated_returns, lag=1)
        assert np.all(result.data.adjacency >= 0)

    def test_threshold_filters(self, block_correlated_returns: pd.DataFrame) -> None:
        low = build_transfer_entropy_network(block_correlated_returns, threshold=0.0)
        high = build_transfer_entropy_network(block_correlated_returns, threshold=0.5)
        assert high.data.graph.number_of_edges() <= low.data.graph.number_of_edges()


# ---------------------------------------------------------------------------
# Centrality Metrics
# ---------------------------------------------------------------------------


class TestCentrality:
    def test_metrics_populated(self, block_correlated_returns: pd.DataFrame) -> None:
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.05)
        net = compute_centrality(result.data)
        assert net.metrics is not None
        assert isinstance(net.metrics, NetworkMetrics)
        assert set(net.metrics.degree.keys()) == set(net.node_names)
        assert set(net.metrics.pagerank.keys()) == set(net.node_names)

    def test_pagerank_sums_to_one(self, block_correlated_returns: pd.DataFrame) -> None:
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.05)
        net = compute_centrality(result.data)
        assert net.metrics is not None
        total_pr = sum(net.metrics.pagerank.values())
        assert_allclose(total_pr, 1.0, atol=1e-6)

    def test_betweenness_bounded(self, block_correlated_returns: pd.DataFrame) -> None:
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.05)
        net = compute_centrality(result.data)
        assert net.metrics is not None
        for v in net.metrics.betweenness.values():
            assert 0 <= v <= 1.0

    def test_preserves_graph(self, block_correlated_returns: pd.DataFrame) -> None:
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.05)
        net = compute_centrality(result.data)
        assert net.graph is result.data.graph
        assert net.method == result.data.method


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------


class TestValidation:
    def test_insufficient_assets(self, rng: np.random.Generator) -> None:
        from quantspt.errors import SPTInvariantError

        with pytest.raises(SPTInvariantError, match="at least 2 assets"):
            build_partial_correlation_network(rng.standard_normal((100, 1)), alpha=0.1)

    def test_nan_returns_rejected(self) -> None:
        from quantspt.errors import SPTInvariantError

        data = np.array([[1.0, 2.0], [np.nan, 3.0]])
        with pytest.raises(SPTInvariantError, match="NaN"):
            build_partial_correlation_network(data, alpha=0.1)

    def test_1d_returns_rejected(self) -> None:
        from quantspt.errors import SPTInvariantError

        with pytest.raises(SPTInvariantError, match="2-D"):
            build_partial_correlation_network(np.array([1.0, 2.0, 3.0]), alpha=0.1)


# ---------------------------------------------------------------------------
# Regression tests for critical bug fixes
# ---------------------------------------------------------------------------


class TestBetweennessUsesDistance:
    """Betweenness centrality must use distance = 1/weight, not raw weight.

    NetworkX interprets the ``weight`` parameter as cost/distance for
    shortest-path computation.  Financial edge weights are connection
    STRENGTH (higher = stronger), so passing them as-is inverts the
    semantics: it treats strong links as long/costly paths.

    The fix: create ``distance = 1 / weight`` and pass
    ``weight='distance'`` to ``nx.betweenness_centrality``.
    """

    def test_bridge_node_has_highest_betweenness(self) -> None:
        """Node C bridges two cliques; it must have highest betweenness."""
        import networkx as nx

        G = nx.Graph()
        G.add_edge("A", "B", weight=5.0)
        G.add_edge("A", "C", weight=5.0)
        G.add_edge("B", "C", weight=5.0)
        G.add_edge("C", "D", weight=5.0)
        G.add_edge("D", "E", weight=5.0)
        G.add_edge("C", "E", weight=5.0)

        adj = np.zeros((5, 5))
        names = ["A", "B", "C", "D", "E"]
        net = FinancialNetwork(graph=G, adjacency=adj, node_names=names, method="test")
        result = compute_centrality(net)
        assert result.metrics is not None
        bc = result.metrics.betweenness
        assert bc["C"] == max(bc.values()), (
            f"Bridge node C should have highest betweenness, got {bc}"
        )

    def test_strong_weight_means_short_distance(self) -> None:
        """A→B with weight 10 should be preferred over A→C→B with weights 1."""
        import networkx as nx

        G = nx.Graph()
        G.add_edge("A", "B", weight=10.0)
        G.add_edge("A", "C", weight=1.0)
        G.add_edge("C", "B", weight=1.0)

        adj = np.zeros((3, 3))
        net = FinancialNetwork(
            graph=G, adjacency=adj, node_names=["A", "B", "C"], method="test"
        )
        result = compute_centrality(net)
        assert result.metrics is not None
        assert result.metrics.betweenness["C"] == 0.0, (
            "C should not be on shortest path when A-B has strong direct link"
        )


class TestGlassoUndirected:
    """GLASSO partial correlations are symmetric; the graph must be undirected."""

    def test_glasso_graph_is_undirected(
        self, block_correlated_returns: pd.DataFrame
    ) -> None:
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.05)
        assert not result.data.graph.is_directed(), (
            "GLASSO network must be undirected (partial correlations are symmetric)"
        )


class TestDirectedDegrees:
    """Directed networks must report in-degree and out-degree separately."""

    def test_granger_has_in_out_degree(self, granger_returns: pd.DataFrame) -> None:
        result = build_granger_network(granger_returns, max_lag=3, significance=0.1)
        net = compute_centrality(result.data)
        assert net.metrics is not None
        assert net.metrics.in_degree is not None, "Directed graph must have in_degree"
        assert net.metrics.out_degree is not None, "Directed graph must have out_degree"

    def test_undirected_has_no_in_out(
        self, block_correlated_returns: pd.DataFrame
    ) -> None:
        result = build_partial_correlation_network(block_correlated_returns, alpha=0.05)
        net = compute_centrality(result.data)
        assert net.metrics is not None
        assert net.metrics.in_degree is None
        assert net.metrics.out_degree is None
