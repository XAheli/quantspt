"""Causal factor models for asset returns.

Learns a causal DAG over factors and assets, then extracts factor
loadings from the structural equation model.  Unlike standard PCA-based
factor models, the causal factor model distinguishes *causal* factors
(parents in the DAG) from spurious correlates.

The key decomposition mirrors the linear SEM:

    X = B X + ε,   ε ~ N(0, Ω)

where B is the causal loading matrix and Ω is the diagonal noise
covariance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._preconditions import require

if TYPE_CHECKING:
    from pgmpy.models import LinearGaussianBayesianNetwork

__all__ = [
    "CausalFactorModel",
]


class CausalFactorModel:
    """Factor model grounded in causal DAG structure.

    Parameters
    ----------
    edges : list of (str, str) or None
        Directed edges ``(parent, child)`` defining the factor–asset
        DAG.  If None, must be supplied at ``.fit()`` time.
    factor_names : list of str or None
        Names of variables to treat as factors.  If None, factors are
        inferred as root nodes (nodes with no parents) of the DAG.
    """

    def __init__(
        self,
        edges: list[tuple[str, str]] | None = None,
        factor_names: list[str] | None = None,
    ) -> None:
        self._edges = edges
        self._factor_names_init = factor_names
        self._fitted = False

        self._model: LinearGaussianBayesianNetwork | None = None
        self._variable_names: list[str] = []
        self._factor_names: list[str] = []
        self._asset_names: list[str] = []
        self._B: NDArray[np.float64] | None = None
        self._omega: NDArray[np.float64] | None = None

    def fit(
        self,
        data: pd.DataFrame | NDArray[np.float64],
        *,
        edges: list[tuple[str, str]] | None = None,
        variable_names: list[str] | None = None,
        factor_names: list[str] | None = None,
        **kwargs: Any,
    ) -> CausalFactorModel:
        """Fit the causal factor model.

        Parameters
        ----------
        data : DataFrame or ndarray of shape (T, p)
            Observational data containing both factor and asset columns.
        edges : list of (str, str), optional
            DAG edges.  Overrides constructor argument.
        variable_names : list of str, optional
            Required when *data* is an ndarray.
        factor_names : list of str, optional
            Override the factor names set at construction.
        **kwargs
            Forwarded to ``model.fit()``.

        Returns
        -------
        CausalFactorModel
            The fitted model (for chaining).
        """
        from pgmpy.models import LinearGaussianBayesianNetwork

        resolved_edges = edges if edges is not None else self._edges
        require(
            resolved_edges is not None and len(resolved_edges) > 0,
            "edges must be supplied either at init or at fit time",
        )
        assert resolved_edges is not None

        df = self._to_dataframe(data, variable_names)
        self._variable_names = sorted(df.columns)

        model = LinearGaussianBayesianNetwork(resolved_edges)
        model.fit(df, **kwargs)
        self._model = model

        resolved_factors = (
            factor_names if factor_names is not None else self._factor_names_init
        )
        if resolved_factors is not None:
            self._factor_names = sorted(resolved_factors)
        else:
            parents_in_dag = {p for p, _ in resolved_edges}
            children_in_dag = {c for _, c in resolved_edges}
            roots = parents_in_dag - children_in_dag
            self._factor_names = sorted(roots)

        self._asset_names = sorted(set(self._variable_names) - set(self._factor_names))
        self._B, self._omega = self._extract_loadings()
        self._fitted = True
        return self

    @classmethod
    def from_pgmpy(
        cls,
        model: LinearGaussianBayesianNetwork,
        factor_names: list[str] | None = None,
    ) -> CausalFactorModel:
        """Wrap a user-provided fitted ``LinearGaussianBayesianNetwork``.

        Parameters
        ----------
        model : LinearGaussianBayesianNetwork
            An already-fitted pgmpy model.
        factor_names : list of str, optional
            If None, root nodes are treated as factors.

        Returns
        -------
        CausalFactorModel
        """
        obj = cls.__new__(cls)
        obj._edges = list(model.edges())
        obj._factor_names_init = factor_names
        obj._model = model
        obj._variable_names = sorted(model.nodes())

        if factor_names is not None:
            obj._factor_names = sorted(factor_names)
        else:
            parents = {p for p, _ in model.edges()}
            children = {c for _, c in model.edges()}
            obj._factor_names = sorted(parents - children)

        obj._asset_names = sorted(set(obj._variable_names) - set(obj._factor_names))
        obj._B, obj._omega = obj._extract_loadings()
        obj._fitted = True
        return obj

    def causal_loadings(self) -> NDArray[np.float64]:
        r"""Full causal loading matrix B ∈ ℝ^{p×p}.

        ``B[i, j]`` is the direct causal effect of variable *j* on
        variable *i*.  Variables are in ``variable_names`` order.

        Returns
        -------
        ndarray of shape (p, p)
        """
        require(self._fitted, "Must call .fit() first")
        assert self._B is not None
        return self._B.copy()

    def factor_loadings(self) -> NDArray[np.float64]:
        """Sub-matrix of B: assets × factors.

        Returns
        -------
        ndarray of shape (n_assets, n_factors)
            Each row corresponds to an asset, each column to a factor.
        """
        require(self._fitted, "Must call .fit() first")
        assert self._B is not None

        node_idx = {n: i for i, n in enumerate(self._variable_names)}
        asset_idx = [node_idx[a] for a in self._asset_names]
        factor_idx = [node_idx[f] for f in self._factor_names]
        return self._B[np.ix_(asset_idx, factor_idx)].copy()

    def noise_covariance(self) -> NDArray[np.float64]:
        """Diagonal noise variances Ω.

        Returns
        -------
        ndarray of shape (p,)
            One variance per variable in ``variable_names`` order.
        """
        require(self._fitted, "Must call .fit() first")
        assert self._omega is not None
        return self._omega.copy()

    def reconstruct_covariance(self) -> NDArray[np.float64]:
        r"""Reconstruct Σ = (I − B)⁻¹ Ω (I − B)⁻ᵀ.

        Returns
        -------
        ndarray of shape (p, p)
            Full covariance matrix.
        """
        require(self._fitted, "Must call .fit() first")
        assert self._B is not None and self._omega is not None

        n = len(self._variable_names)
        I_minus_B_inv = np.linalg.inv(np.eye(n) - self._B)
        return I_minus_B_inv @ np.diag(self._omega) @ I_minus_B_inv.T

    @property
    def variable_names(self) -> list[str]:
        """All variable names in sorted order."""
        require(self._fitted, "Must call .fit() first")
        return list(self._variable_names)

    @property
    def factor_names(self) -> list[str]:
        """Factor (parent/root) variable names."""
        require(self._fitted, "Must call .fit() first")
        return list(self._factor_names)

    @property
    def asset_names(self) -> list[str]:
        """Asset (non-factor) variable names."""
        require(self._fitted, "Must call .fit() first")
        return list(self._asset_names)

    def _extract_loadings(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert self._model is not None
        n = len(self._variable_names)
        node_idx = {name: i for i, name in enumerate(self._variable_names)}

        B = np.zeros((n, n), dtype=np.float64)
        omega = np.zeros(n, dtype=np.float64)

        for cpd in self._model.get_cpds():
            child_idx = node_idx[cpd.variable]
            omega[child_idx] = cpd.std**2
            for k, parent in enumerate(cpd.evidence):
                parent_idx = node_idx[parent]
                B[child_idx, parent_idx] = cpd.beta[k + 1]

        return B, omega

    @staticmethod
    def _to_dataframe(
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
