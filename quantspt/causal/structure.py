"""Causal structure learning with pluggable discovery algorithms.

Wraps pgmpy's causal discovery suite (PC, GES, HillClimbSearch) behind
a unified, fully-configurable interface.  Users may also bring their own
fitted pgmpy model via the ``from_pgmpy`` classmethod.

All algorithm choices, scoring methods, CI tests, and hyperparameters
are exposed as constructor parameters — nothing is hardcoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._preconditions import require

if TYPE_CHECKING:
    from pgmpy.models import LinearGaussianBayesianNetwork

__all__ = [
    "CausalStructureLearner",
]

_METHODS = frozenset({"pc", "ges", "hillclimb"})

_SCORING_METHODS = frozenset({"bic-g", "aic-g", "ll-g"})

_CI_TESTS = frozenset({"pearsonr", "chi_square", "g_sq"})


class CausalStructureLearner:
    """Learn causal DAG structure from observational data.

    Parameters
    ----------
    method : ``"pc"`` | ``"ges"`` | ``"hillclimb"``
        Discovery algorithm.
    scoring_method : str
        Scoring criterion for score-based methods (GES, HillClimb).
        Passed directly to the pgmpy constructor.
    ci_test : str
        Conditional-independence test for constraint-based methods (PC).
        Passed directly to the pgmpy constructor.
    significance_level : float
        Significance level for CI tests.
    max_cond_vars : int or None
        Maximum conditioning set size for the PC algorithm.
    max_indegree : int or None
        Maximum in-degree per node for score-based methods.
    return_type : str
        ``"dag"`` (default) or ``"pdag"``.  ``"dag"`` avoids
        undirected edges in the output.
    prior_edges : list of (str, str) or None
        Edges that must appear in the learned graph (expert knowledge).
    forbidden_edges : list of (str, str) or None
        Edges that must not appear in the learned graph.
    temporal_order : list of list of str or None
        Temporal tiers for time-ordering constraints.
    **kwargs
        Forwarded to the underlying pgmpy discovery constructor.
    """

    def __init__(
        self,
        method: Literal["pc", "ges", "hillclimb"] = "pc",
        *,
        scoring_method: str = "bic-g",
        ci_test: str = "pearsonr",
        significance_level: float = 0.05,
        max_cond_vars: int | None = None,
        max_indegree: int | None = None,
        return_type: str = "dag",
        prior_edges: list[tuple[str, str]] | None = None,
        forbidden_edges: list[tuple[str, str]] | None = None,
        temporal_order: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        require(method in _METHODS, f"method must be one of {_METHODS}, got {method!r}")
        self._method = method
        self._scoring_method = scoring_method
        self._ci_test = ci_test
        self._significance_level = significance_level
        self._max_cond_vars = max_cond_vars
        self._max_indegree = max_indegree
        self._return_type = return_type
        self._prior_edges = prior_edges
        self._forbidden_edges = forbidden_edges
        self._temporal_order = temporal_order
        self._extra_kwargs = kwargs

        self._fitted = False
        self._adjacency_matrix: NDArray[np.float64] | None = None
        self._edges: list[tuple[str, str]] = []
        self._variable_names: list[str] = []
        self._discovery_object: Any = None

    def fit(
        self,
        data: pd.DataFrame | NDArray[np.float64],
        *,
        variable_names: list[str] | None = None,
        **kwargs: Any,
    ) -> CausalStructureLearner:
        """Learn causal structure from observational data.

        Parameters
        ----------
        data : DataFrame or ndarray of shape (T, p)
            Observational data.  If an ndarray, *variable_names* must
            be supplied.
        variable_names : list of str, optional
            Column names when *data* is an ndarray.
        **kwargs
            Merged with constructor kwargs and forwarded to the pgmpy
            discovery object.

        Returns
        -------
        CausalStructureLearner
            The fitted learner (for chaining).
        """
        df = self._to_dataframe(data, variable_names)
        self._variable_names = list(df.columns)

        merged = {**self._extra_kwargs, **kwargs}
        discovery = self._build_discovery(merged)

        expert = self._build_expert_knowledge()
        if expert is not None:
            discovery.fit(df, expert_knowledge=expert)
        else:
            discovery.fit(df)

        self._discovery_object = discovery
        self._adjacency_matrix = np.array(discovery.adjacency_matrix_, dtype=np.float64)
        self._edges = list(discovery.causal_graph_.edges())
        self._fitted = True
        return self

    @property
    def adjacency_matrix(self) -> NDArray[np.float64]:
        """Binary adjacency matrix of the learned DAG."""
        require(self._fitted, "Must call .fit() before accessing adjacency_matrix")
        assert self._adjacency_matrix is not None
        return self._adjacency_matrix

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Directed edges ``(parent, child)`` of the learned DAG."""
        require(self._fitted, "Must call .fit() before accessing edges")
        return list(self._edges)

    @property
    def variable_names(self) -> list[str]:
        """Variable names in the order used by the adjacency matrix."""
        require(self._fitted, "Must call .fit() before accessing variable_names")
        return list(self._variable_names)

    @classmethod
    def from_pgmpy(
        cls,
        model: LinearGaussianBayesianNetwork,
    ) -> CausalStructureLearner:
        """Create a learner from a user-provided fitted pgmpy model.

        Parameters
        ----------
        model : LinearGaussianBayesianNetwork
            A fitted pgmpy model.  The caller is responsible for
            ensuring the model is valid.

        Returns
        -------
        CausalStructureLearner
            A learner whose ``edges`` and ``adjacency_matrix`` are
            derived from the supplied model.
        """
        obj = cls.__new__(cls)
        obj._method = "external"  # type: ignore[assignment]
        obj._scoring_method = ""
        obj._ci_test = ""
        obj._significance_level = 0.05
        obj._max_cond_vars = None
        obj._max_indegree = None
        obj._return_type = "dag"
        obj._prior_edges = None
        obj._forbidden_edges = None
        obj._temporal_order = None
        obj._extra_kwargs = {}
        obj._discovery_object = None

        nodes = sorted(model.nodes())
        obj._variable_names = nodes
        obj._edges = list(model.edges())

        n = len(nodes)
        idx = {node: i for i, node in enumerate(nodes)}
        adj = np.zeros((n, n), dtype=np.float64)
        for parent, child in model.edges():
            adj[idx[parent], idx[child]] = 1.0
        obj._adjacency_matrix = adj
        obj._fitted = True
        return obj

    def _to_dataframe(
        self,
        data: pd.DataFrame | NDArray[np.float64],
        variable_names: list[str] | None,
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        require(
            variable_names is not None,
            "variable_names required when data is an ndarray",
        )
        assert variable_names is not None
        return pd.DataFrame(data, columns=variable_names)

    def _build_discovery(self, extra: dict[str, Any]) -> Any:
        from pgmpy.causal_discovery import GES, PC, HillClimbSearch

        if self._method == "pc":
            init_kwargs: dict[str, Any] = {
                "ci_test": self._ci_test,
                "significance_level": self._significance_level,
                "return_type": self._return_type,
            }
            if self._max_cond_vars is not None:
                init_kwargs["max_cond_vars"] = self._max_cond_vars
            init_kwargs.update(extra)
            return PC(**init_kwargs)

        if self._method == "ges":
            init_kwargs = {
                "scoring_method": self._scoring_method,
            }
            init_kwargs.update(extra)
            return GES(**init_kwargs)

        init_kwargs = {
            "scoring_method": self._scoring_method,
        }
        if self._max_indegree is not None:
            init_kwargs["max_indegree"] = self._max_indegree
        init_kwargs.update(extra)
        return HillClimbSearch(**init_kwargs)

    def _build_expert_knowledge(self) -> Any | None:
        if (
            self._prior_edges is None
            and self._forbidden_edges is None
            and self._temporal_order is None
        ):
            return None

        from pgmpy.causal_discovery import ExpertKnowledge

        kwargs: dict[str, Any] = {}
        if self._prior_edges is not None:
            kwargs["prior_edges"] = self._prior_edges
        if self._forbidden_edges is not None:
            kwargs["forbidden_edges"] = self._forbidden_edges
        if self._temporal_order is not None:
            kwargs["temporal_order"] = self._temporal_order

        return ExpertKnowledge(**kwargs)
