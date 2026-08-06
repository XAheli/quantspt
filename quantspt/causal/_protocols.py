"""Causal inference protocols for quantspt.

Defines structural typing contracts for causal models, ensuring any
compliant implementation (pgmpy-backed or user-provided) integrates
seamlessly with quantspt's covariance and factor model pipeline.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self


@runtime_checkable
class CausalStructureModel(Protocol):
    """Protocol for causal structure learning algorithms.

    Any object satisfying this interface can be used as the backbone
    of ``CausalStructureLearner``.  The pgmpy discovery objects (PC,
    GES, HillClimbSearch) already satisfy this contract after
    ``.fit()``.
    """

    def fit(
        self,
        data: Any,
        *,
        variable_names: list[str] | None = None,
        prior_edges: list[tuple[str, str]] | None = None,
        forbidden_edges: list[tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Learn causal structure from observational data.

        Parameters
        ----------
        data : DataFrame or ndarray
            Observational data with variables as columns.
        variable_names : list of str, optional
            Column names when *data* is an ndarray.
        prior_edges : list of (str, str), optional
            Edges that must appear in the learned graph.
        forbidden_edges : list of (str, str), optional
            Edges that must not appear in the learned graph.
        **kwargs
            Algorithm-specific parameters.

        Returns
        -------
        Self
        """
        ...

    @property
    def adjacency_matrix(self) -> NDArray[np.float64]:
        """Adjacency matrix of the learned DAG.

        Returns
        -------
        ndarray of shape (p, p)
            Binary matrix where entry (i, j) = 1 indicates an edge
            from variable i to variable j.
        """
        ...

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Directed edges of the learned DAG.

        Returns
        -------
        list of (str, str)
            Each tuple ``(parent, child)`` represents a directed edge.
        """
        ...


@runtime_checkable
class CausalCovarianceModel(Protocol):
    """Protocol for causal covariance estimators.

    Implementations produce observational and interventional covariance
    matrices from a fitted linear Gaussian Bayesian network.
    """

    def fit(
        self,
        data: Any,
        *,
        edges: list[tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the causal covariance model.

        Parameters
        ----------
        data : DataFrame
            Observational data.
        edges : list of (str, str), optional
            DAG edges.  If None, uses a previously learned structure.
        **kwargs
            Model-specific parameters.

        Returns
        -------
        Self
        """
        ...

    def observational_covariance(self) -> NDArray[np.float64]:
        """Observational covariance Σ.

        Returns
        -------
        ndarray of shape (p, p)
            Symmetric PSD covariance matrix under no interventions.
        """
        ...

    def interventional_covariance(
        self,
        interventions: dict[str, float],
    ) -> NDArray[np.float64]:
        """Interventional covariance Σ_do.

        Parameters
        ----------
        interventions : dict mapping variable name → fixed value
            Variables subjected to a do-intervention.

        Returns
        -------
        ndarray of shape (p, p)
            Covariance matrix under the specified interventions.
        """
        ...


@runtime_checkable
class CausalFactorModelProtocol(Protocol):
    """Protocol for causal factor models.

    Extracts causal loadings (B matrix) and noise covariance (Ω) from
    a fitted linear Gaussian Bayesian network, distinguishing causal
    factors from spurious correlates.
    """

    def fit(
        self,
        data: Any,
        *,
        edges: list[tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the causal factor model.

        Parameters
        ----------
        data : DataFrame
            Observational data with variables as columns.
        edges : list of (str, str), optional
            DAG edges defining the causal structure.
        **kwargs
            Model-specific parameters.

        Returns
        -------
        Self
        """
        ...

    def causal_loadings(self) -> NDArray[np.float64]:
        """Causal loading matrix B.

        Returns
        -------
        ndarray of shape (p, p)
            Entry B[i, j] is the direct causal effect of variable j
            on variable i.
        """
        ...

    def noise_covariance(self) -> NDArray[np.float64]:
        """Diagonal noise covariance Ω.

        Returns
        -------
        ndarray of shape (p,)
            Noise (residual) variance for each variable.
        """
        ...


__all__ = [
    "CausalCovarianceModel",
    "CausalFactorModelProtocol",
    "CausalStructureModel",
]
