"""Tests for causal structure learning.

Validates DAG recovery from synthetic data with KNOWN causal structure,
algorithm pluggability, and ExpertKnowledge integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pgmpy")

from quantspt.causal.structure import CausalStructureLearner
from quantspt.errors import SPTInvariantError


@pytest.fixture()
def chain_data() -> pd.DataFrame:
    """Synthetic data from X → Y → Z (chain graph)."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(size=n)
    Y = 0.8 * X + rng.normal(size=n) * 0.3
    Z = 0.5 * Y + rng.normal(size=n) * 0.2
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


@pytest.fixture()
def fork_data() -> pd.DataFrame:
    """Synthetic data from a fork: X → Y, X → Z."""
    rng = np.random.default_rng(123)
    n = 500
    X = rng.normal(size=n)
    Y = 0.7 * X + rng.normal(size=n) * 0.4
    Z = 0.6 * X + rng.normal(size=n) * 0.3
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


class TestCausalStructureLearnerPC:
    """PC algorithm discovery tests."""

    def test_pc_recovers_chain_edges(self, chain_data: pd.DataFrame) -> None:
        learner = CausalStructureLearner(
            method="pc", ci_test="pearsonr", significance_level=0.05
        )
        learner.fit(chain_data)
        edges = learner.edges
        edge_set = {frozenset(e) for e in edges}
        assert frozenset(("X", "Y")) in edge_set
        assert frozenset(("Y", "Z")) in edge_set

    def test_pc_adjacency_matrix_shape(self, chain_data: pd.DataFrame) -> None:
        learner = CausalStructureLearner(method="pc")
        learner.fit(chain_data)
        adj = learner.adjacency_matrix
        assert adj.shape == (3, 3)
        assert adj.dtype == np.float64

    def test_pc_adjacency_binary(self, chain_data: pd.DataFrame) -> None:
        learner = CausalStructureLearner(method="pc")
        learner.fit(chain_data)
        adj = learner.adjacency_matrix
        assert set(np.unique(adj)).issubset({0.0, 1.0})

    def test_pc_variable_names(self, chain_data: pd.DataFrame) -> None:
        learner = CausalStructureLearner(method="pc")
        learner.fit(chain_data)
        assert set(learner.variable_names) == {"X", "Y", "Z"}

    def test_pc_no_spurious_edge(self, chain_data: pd.DataFrame) -> None:
        """X and Z are d-separated given Y, so no direct X-Z edge."""
        learner = CausalStructureLearner(
            method="pc", ci_test="pearsonr", significance_level=0.01
        )
        learner.fit(chain_data)
        edges = learner.edges
        direct_xz = any(
            (a == "X" and b == "Z") or (a == "Z" and b == "X") for a, b in edges
        )
        assert not direct_xz

    def test_pc_fork_structure(self, fork_data: pd.DataFrame) -> None:
        learner = CausalStructureLearner(method="pc")
        learner.fit(fork_data)
        edges = learner.edges
        edge_set = {frozenset(e) for e in edges}
        assert frozenset(("X", "Y")) in edge_set
        assert frozenset(("X", "Z")) in edge_set

    def test_pc_configurable_ci_test(self, chain_data: pd.DataFrame) -> None:
        """pearsonr is the natural choice for continuous data;
        chi_square and g_sq are for discrete data and may find zero
        edges on continuous inputs, so we only assert they run without
        error."""
        learner = CausalStructureLearner(method="pc", ci_test="pearsonr")
        learner.fit(chain_data)
        assert len(learner.edges) > 0

        for ci in ("chi_square", "g_sq"):
            learner = CausalStructureLearner(method="pc", ci_test=ci)
            learner.fit(chain_data)
            assert learner.adjacency_matrix.shape == (3, 3)

    def test_pc_return_type_dag(self, chain_data: pd.DataFrame) -> None:
        learner = CausalStructureLearner(method="pc", return_type="dag")
        learner.fit(chain_data)
        assert len(learner.edges) >= 2


class TestCausalStructureLearnerGES:
    """GES algorithm discovery tests."""

    def test_ges_recovers_edges(self, chain_data: pd.DataFrame) -> None:
        learner = CausalStructureLearner(method="ges", scoring_method="bic-g")
        learner.fit(chain_data)
        edge_set = {frozenset(e) for e in learner.edges}
        assert frozenset(("X", "Y")) in edge_set
        assert frozenset(("Y", "Z")) in edge_set

    def test_ges_scoring_methods(self, chain_data: pd.DataFrame) -> None:
        for score in ("bic-g", "aic-g"):
            learner = CausalStructureLearner(method="ges", scoring_method=score)
            learner.fit(chain_data)
            assert len(learner.edges) > 0


class TestCausalStructureLearnerHillClimb:
    """HillClimb algorithm tests."""

    def test_hillclimb_recovers_edges(self, chain_data: pd.DataFrame) -> None:
        learner = CausalStructureLearner(method="hillclimb", scoring_method="bic-g")
        learner.fit(chain_data)
        edge_set = {frozenset(e) for e in learner.edges}
        assert frozenset(("X", "Y")) in edge_set

    def test_hillclimb_max_indegree(self, chain_data: pd.DataFrame) -> None:
        learner = CausalStructureLearner(
            method="hillclimb", max_indegree=2, scoring_method="bic-g"
        )
        learner.fit(chain_data)
        adj = learner.adjacency_matrix
        in_degrees = adj.sum(axis=0)
        assert np.all(in_degrees <= 2)


class TestExpertKnowledge:
    """ExpertKnowledge integration tests."""

    def test_forbidden_edges(self, chain_data: pd.DataFrame) -> None:
        learner = CausalStructureLearner(
            method="pc",
            forbidden_edges=[("X", "Y")],
        )
        learner.fit(chain_data)
        assert ("X", "Y") not in learner.edges

    def test_prior_edges(self, chain_data: pd.DataFrame) -> None:
        """Prior edges guarantee the edge exists (possibly reversed)."""
        learner = CausalStructureLearner(
            method="pc",
            prior_edges=[("X", "Y")],
        )
        learner.fit(chain_data)
        edge_set = {frozenset(e) for e in learner.edges}
        assert frozenset(("X", "Y")) in edge_set

    def test_temporal_order(self, chain_data: pd.DataFrame) -> None:
        learner = CausalStructureLearner(
            method="pc",
            temporal_order=[["X"], ["Y"], ["Z"]],
        )
        learner.fit(chain_data)
        for parent, child in learner.edges:
            parent_tier = next(
                i for i, tier in enumerate([["X"], ["Y"], ["Z"]]) if parent in tier
            )
            child_tier = next(
                i for i, tier in enumerate([["X"], ["Y"], ["Z"]]) if child in tier
            )
            assert parent_tier <= child_tier


class TestFromPgmpy:
    """Test from_pgmpy classmethod for user-provided models."""

    def test_from_pgmpy_model(self, chain_data: pd.DataFrame) -> None:
        from pgmpy.models import LinearGaussianBayesianNetwork

        model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
        model.fit(chain_data)

        learner = CausalStructureLearner.from_pgmpy(model)
        assert learner.edges == list(model.edges())
        assert learner.adjacency_matrix.shape == (3, 3)

    def test_from_pgmpy_adjacency_correct(self, chain_data: pd.DataFrame) -> None:
        from pgmpy.models import LinearGaussianBayesianNetwork

        model = LinearGaussianBayesianNetwork([("X", "Y"), ("Y", "Z")])
        model.fit(chain_data)

        learner = CausalStructureLearner.from_pgmpy(model)
        adj = learner.adjacency_matrix
        names = learner.variable_names
        idx = {n: i for i, n in enumerate(names)}

        assert adj[idx["X"], idx["Y"]] == 1.0
        assert adj[idx["Y"], idx["Z"]] == 1.0
        assert adj[idx["X"], idx["Z"]] == 0.0


class TestNdarrayInput:
    """Test ndarray input with explicit variable_names."""

    def test_ndarray_with_names(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(size=n)
        Y = 0.8 * X + rng.normal(size=n) * 0.3
        data = np.column_stack([X, Y])

        learner = CausalStructureLearner(method="pc")
        learner.fit(data, variable_names=["X", "Y"])
        assert len(learner.edges) > 0

    def test_ndarray_without_names_raises(self) -> None:
        data = np.random.default_rng(42).normal(size=(100, 3))
        learner = CausalStructureLearner(method="pc")
        with pytest.raises(SPTInvariantError, match="variable_names"):
            learner.fit(data)


class TestValidation:
    """Input validation tests."""

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="method"):
            CausalStructureLearner(method="invalid")  # type: ignore[arg-type]

    def test_access_before_fit_raises(self) -> None:
        learner = CausalStructureLearner(method="pc")
        with pytest.raises(SPTInvariantError, match="fit"):
            _ = learner.adjacency_matrix
        with pytest.raises(SPTInvariantError, match="fit"):
            _ = learner.edges
        with pytest.raises(SPTInvariantError, match="fit"):
            _ = learner.variable_names
