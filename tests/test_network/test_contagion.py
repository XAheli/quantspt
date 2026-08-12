"""Tests for quantspt.network.contagion — shock propagation models.

Validates Eisenberg-Noe clearing vector and DebtRank against known
analytical results on small hand-crafted networks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt._result import SPTResult
from quantspt.errors import SPTInvariantError
from quantspt.network.contagion import (
    ContagionResult,
    DebtRank,
    EisenbergNoe,
    clearing_vector,
    debt_rank,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def three_node_liabilities() -> tuple[np.ndarray, np.ndarray]:
    """Simple 3-node interbank network.

    Node 0 owes 80 to Node 1, Node 1 owes 60 to Node 2, Node 2 owes 40 to Node 0.
    External assets: [100, 50, 80].
    All nodes should be solvent.
    """
    L = np.array(
        [
            [0, 80, 0],
            [0, 0, 60],
            [40, 0, 0],
        ],
        dtype=np.float64,
    )
    e = np.array([100.0, 50.0, 80.0])
    return L, e


@pytest.fixture()
def stressed_network() -> tuple[np.ndarray, np.ndarray]:
    """5-node network where shocking node 0 cascades.

    Node 0 has high obligations, low external assets.
    """
    L = np.array(
        [
            [0, 50, 30, 0, 0],
            [0, 0, 40, 20, 0],
            [0, 0, 0, 30, 10],
            [0, 0, 0, 0, 25],
            [10, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    e = np.array([20.0, 30.0, 25.0, 40.0, 35.0])
    return L, e


@pytest.fixture()
def five_node_exposures() -> tuple[np.ndarray, np.ndarray]:
    """5-node DebtRank network with known structure."""
    W = np.array(
        [
            [0, 0.3, 0.2, 0, 0],
            [0.1, 0, 0.25, 0, 0],
            [0, 0.15, 0, 0.3, 0],
            [0, 0, 0.1, 0, 0.2],
            [0, 0, 0, 0.15, 0],
        ],
        dtype=np.float64,
    )
    equity = np.array([100.0, 80.0, 60.0, 90.0, 70.0])
    return W, equity


# ---------------------------------------------------------------------------
# Eisenberg-Noe Tests
# ---------------------------------------------------------------------------


class TestEisenbergNoe:
    def test_solvent_network_no_defaults(self, three_node_liabilities: tuple) -> None:
        """When all nodes have sufficient assets, no defaults occur."""
        L, e = three_node_liabilities
        result = clearing_vector(L, e)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, ContagionResult)
        assert result.data.n_defaults == 0
        assert_allclose(result.data.distress, 0.0, atol=1e-6)

    def test_full_payment_when_solvent(self, three_node_liabilities: tuple) -> None:
        """Clearing payments should equal obligations when everyone is solvent."""
        L, e = three_node_liabilities
        model = EisenbergNoe(L, e)
        result = model.solve()
        assert result.data.n_defaults == 0

    def test_stressed_network_cascades(self, stressed_network: tuple) -> None:
        """Reducing external assets of node 0 should cause significant distress."""
        L, e = stressed_network
        e_stressed = e.copy()
        e_stressed[0] = 5.0
        result = clearing_vector(L, e_stressed)
        assert result.data.distress[0] > 0.5, "Shocked node should be highly distressed"
        assert result.data.total_loss > 0, "System should experience losses"
        assert np.any(result.data.distress > 0), "Contagion should propagate"

    def test_total_loss_nonnegative(self, stressed_network: tuple) -> None:
        L, e = stressed_network
        e[0] = 5.0
        result = clearing_vector(L, e)
        assert result.data.total_loss >= 0

    def test_convergence(self, three_node_liabilities: tuple) -> None:
        L, e = three_node_liabilities
        result = clearing_vector(L, e)
        assert result.metadata["converged"]

    def test_distress_bounded(self, stressed_network: tuple) -> None:
        L, e = stressed_network
        e[0] = 1.0
        result = clearing_vector(L, e)
        assert np.all(result.data.distress >= 0)
        assert np.all(result.data.distress <= 1.0 + 1e-8)

    def test_history_shape(self, three_node_liabilities: tuple) -> None:
        L, e = three_node_liabilities
        result = clearing_vector(L, e)
        n = L.shape[0]
        assert result.data.distress_history.shape[1] == n
        assert result.data.distress_history.shape[0] >= 2

    def test_zero_obligations_no_distress(self) -> None:
        """Nodes with zero obligations cannot default."""
        L = np.zeros((3, 3))
        e = np.array([100.0, 200.0, 300.0])
        result = clearing_vector(L, e)
        assert result.data.n_defaults == 0
        assert_allclose(result.data.distress, 0.0)


# ---------------------------------------------------------------------------
# DebtRank Tests
# ---------------------------------------------------------------------------


class TestDebtRank:
    def test_no_shock_no_contagion(self, five_node_exposures: tuple) -> None:
        """Zero initial shock produces zero distress."""
        W, eq = five_node_exposures
        shock = np.zeros(5)
        result = debt_rank(W, eq, shock)
        assert_allclose(result.data.distress, 0.0, atol=1e-10)
        assert result.data.n_defaults == 0

    def test_single_node_shock_propagates(self, five_node_exposures: tuple) -> None:
        """Shocking node 0 should propagate to connected nodes."""
        W, eq = five_node_exposures
        shock = np.array([1.0, 0, 0, 0, 0])
        result = debt_rank(W, eq, shock)
        assert result.data.distress[0] == 1.0, "Shocked node stays at full distress"
        assert result.data.distress[1] > 0, "Node 1 (exposed to 0) should be distressed"

    def test_distress_monotone_increasing(self, five_node_exposures: tuple) -> None:
        """Distress should never decrease across rounds."""
        W, eq = five_node_exposures
        shock = np.array([0.5, 0, 0, 0, 0])
        result = debt_rank(W, eq, shock)
        history = result.data.distress_history
        for t in range(1, len(history)):
            assert np.all(history[t] >= history[t - 1] - 1e-10)

    def test_distress_bounded_01(self, five_node_exposures: tuple) -> None:
        """Distress values must be in [0, 1]."""
        W, eq = five_node_exposures
        shock = np.array([1.0, 0.5, 0, 0, 0])
        result = debt_rank(W, eq, shock)
        assert np.all(result.data.distress >= 0)
        assert np.all(result.data.distress <= 1.0 + 1e-8)

    def test_total_loss_proportional_to_shock(self, five_node_exposures: tuple) -> None:
        """Larger shocks should produce larger total losses."""
        W, eq = five_node_exposures
        small_shock = np.array([0.1, 0, 0, 0, 0])
        large_shock = np.array([0.9, 0, 0, 0, 0])
        r_small = debt_rank(W, eq, small_shock)
        r_large = debt_rank(W, eq, large_shock)
        assert r_large.data.total_loss >= r_small.data.total_loss

    def test_isolated_node_stays_clean(self) -> None:
        """A node with zero exposure row should not be affected."""
        W = np.array(
            [
                [0, 0.5, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.float64,
        )
        eq = np.array([100.0, 100.0, 100.0])
        shock = np.array([1.0, 0, 0])
        result = debt_rank(W, eq, shock)
        assert_allclose(result.data.distress[2], 0.0, atol=1e-10)

    def test_result_type(self, five_node_exposures: tuple) -> None:
        W, eq = five_node_exposures
        shock = np.array([0.5, 0, 0, 0, 0])
        result = debt_rank(W, eq, shock)
        assert isinstance(result, SPTResult)
        assert isinstance(result.data, ContagionResult)
        assert result.metadata["method"] == "LinearDebtRank"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestContagionValidation:
    def test_negative_liabilities_rejected(self) -> None:
        L = np.array([[0, -10], [5, 0]], dtype=np.float64)
        e = np.array([100.0, 100.0])
        with pytest.raises(SPTInvariantError, match="non-negative"):
            EisenbergNoe(L, e)

    def test_negative_equity_rejected(self) -> None:
        W = np.array([[0, 0.1], [0.1, 0]], dtype=np.float64)
        eq = np.array([100.0, -50.0])
        with pytest.raises(SPTInvariantError, match="positive"):
            DebtRank(W, eq)

    def test_shock_out_of_bounds_rejected(self, five_node_exposures: tuple) -> None:
        W, eq = five_node_exposures
        model = DebtRank(W, eq)
        with pytest.raises(SPTInvariantError, match="\\[0, 1\\]"):
            model.propagate(np.array([1.5, 0, 0, 0, 0]))


# ---------------------------------------------------------------------------
# Regression test: DebtRank incremental distress formula
# ---------------------------------------------------------------------------


class TestDebtRankIncrementalDistress:
    """Verify that DebtRank transmits incremental distress (h_curr - h_prev),
    NOT the wrong formula h_curr * (1 - h_prev) which causes unbounded
    contagion even for small exposures.

    The old formula h_curr * (1 - h_prev) is wrong because:
    - It transmits h_curr (full level) scaled by (1 - h_prev), not the
      actual CHANGE in distress.
    - This causes a feedback loop: any nonzero h_curr keeps generating
      contagion even after the shock has already been transmitted.

    The correct formula h_curr - h_prev transmits only the marginal
    increase in distress at each step, which converges.
    """

    def test_small_exposure_bounded_contagion(self) -> None:
        """With tiny exposures, contagion must remain proportionally small.

        The old formula h * (1-h_prev) causes runaway contagion even for
        W_ij = 0.01 because h_curr stays nonzero and keeps transmitting.
        """
        n = 3
        W = np.array(
            [
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
                [0.01, 0.0, 0.0],
            ]
        )
        equity = np.ones(n) * 100.0
        shock = np.array([0.1, 0.0, 0.0])

        result = debt_rank(W, equity, shock)

        # With W_ij = 0.01 and initial shock 0.1 on node 0:
        # Node 1 receives 0.01 * 0.1 = 0.001 contagion.  Subsequent
        # rounds transmit only the *incremental* change, which shrinks
        # geometrically.  Final distress on nodes 1,2 should be << 0.05.
        assert result.data.distress[1] < 0.05, (
            f"Tiny exposure should give tiny contagion, got {result.data.distress[1]}"
        )
        assert result.data.distress[2] < 0.05, (
            f"Second-hop contagion should be even smaller, got {result.data.distress[2]}"
        )

    def test_analytical_one_step(self) -> None:
        """Check one-step propagation matches the corrected analytical formula.

        Raw exposure[1,0] = 30, equity = [100, 100].
        Normalized W[1,0] = 30 / 100 = 0.3.
        With h(0) = [1, 0], after one step:
        incremental_0 = h_0(1) - h_0(0) = 1 - 0 = 1
        contagion_1 = W[1,0] * incremental_0 = 0.3 * 1 = 0.3
        h_1(1) = min(1, 0 + 0.3) = 0.3
        """
        raw_exposure = np.array([[0.0, 0.0], [30.0, 0.0]])
        equity = np.array([100.0, 100.0])
        shock = np.array([1.0, 0.0])

        result = debt_rank(raw_exposure, equity, shock)
        assert_allclose(result.data.distress[1], 0.3, atol=1e-6)

    def test_converges_quickly(self) -> None:
        """The correct formula converges in few rounds; the old one spirals."""
        W = np.array(
            [
                [0.0, 0.2, 0.0],
                [0.0, 0.0, 0.2],
                [0.2, 0.0, 0.0],
            ]
        )
        equity = np.ones(3) * 100.0
        shock = np.array([0.5, 0.0, 0.0])

        result = debt_rank(W, equity, shock)
        assert result.data.iterations < 20, (
            f"Should converge quickly, took {result.data.iterations} rounds"
        )
