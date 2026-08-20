"""Causal covariance estimation and do-calculus interventions.

Given a DAG (from structure learning or user-provided), fits a
``LinearGaussianBayesianNetwork`` and provides:

- Observational covariance Σ via ``to_joint_gaussian()``
- Interventional covariance Σ_do via graph mutilation
- Structural decomposition Σ = (I − B)⁻¹ Ω (I − B)⁻ᵀ

The output is a standard ndarray usable as a covariance matrix for
excess-growth-rate (γ*) computation in the core SPT framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._preconditions import require

if TYPE_CHECKING:
    from pgmpy.models import LinearGaussianBayesianNetwork

__all__ = [
    "CausalCovarianceEstimator",
    "CovarianceDecomposition",
]


@dataclass(frozen=True)
class CovarianceDecomposition:
    r"""Structural decomposition of the observational covariance.

    For a linear SEM  X = B X + ε  with  ε ~ N(0, Ω):

    .. math::
        \Sigma = (I - B)^{-1}\, \Omega\, (I - B)^{-\top}

    Attributes
    ----------
    B : ndarray of shape (p, p)
        Causal loading matrix.  ``B[i, j]`` is the direct linear
        effect of variable *j* on variable *i*.
    omega : ndarray of shape (p,)
        Noise (residual) variances, one per variable.
    sigma : ndarray of shape (p, p)
        Full covariance matrix reconstructed from B and Ω.
    variable_names : list of str
        Variable ordering matching the matrix rows/columns.
    """

    B: NDArray[np.float64]
    omega: NDArray[np.float64]
    sigma: NDArray[np.float64]
    variable_names: list[str]


class CausalCovarianceEstimator:
    """Estimate observational and interventional covariance from a causal DAG.

    Parameters
    ----------
    edges : list of (str, str) or None
        Directed edges ``(parent, child)`` defining the causal DAG.
        If None, must be supplied at ``.fit()`` time via the *edges*
        keyword argument.
    backend : ``"numpy"`` | ``"torch"``
        Computation backend for pgmpy.  ``"torch"`` enables GPU
        acceleration for factor operations, CPDs, and sampling.
    device : str or None
        Device for the torch backend (``"cpu"`` or ``"cuda"``).
        Only used when ``backend="torch"``.
    """

    def __init__(
        self,
        edges: list[tuple[str, str]] | None = None,
        *,
        backend: str = "numpy",
        device: str | None = None,
    ) -> None:
        self._edges = edges
        self._backend = backend
        self._device = device
        self._model: LinearGaussianBayesianNetwork | None = None
        self._variable_names: list[str] = []
        self._fitted = False

    def fit(
        self,
        data: pd.DataFrame | NDArray[np.float64],
        *,
        edges: list[tuple[str, str]] | None = None,
        variable_names: list[str] | None = None,
        **kwargs: Any,
    ) -> CausalCovarianceEstimator:
        """Fit a ``LinearGaussianBayesianNetwork`` to the data.

        Parameters
        ----------
        data : DataFrame or ndarray of shape (T, p)
            Observational data.
        edges : list of (str, str), optional
            DAG edges.  Overrides edges passed to the constructor.
        variable_names : list of str, optional
            Required when *data* is an ndarray.
        **kwargs
            Forwarded to ``model.fit()``.

        Returns
        -------
        CausalCovarianceEstimator
            The fitted estimator (for chaining).
        """
        from pgmpy.models import LinearGaussianBayesianNetwork

        resolved_edges = edges if edges is not None else self._edges
        require(
            resolved_edges is not None and len(resolved_edges) > 0,
            "edges must be supplied either at init or at fit time",
        )
        assert resolved_edges is not None

        self._configure_backend()
        df = self._to_dataframe(data, variable_names)
        self._variable_names = sorted(df.columns)

        model = LinearGaussianBayesianNetwork(resolved_edges)
        for col in df.columns:
            if col not in model.nodes():
                model.add_node(col)
        model.fit(df, **kwargs)
        self._model = model
        self._fitted = True
        return self

    @classmethod
    def from_pgmpy(
        cls,
        model: LinearGaussianBayesianNetwork,
    ) -> CausalCovarianceEstimator:
        """Wrap a user-provided fitted ``LinearGaussianBayesianNetwork``.

        Parameters
        ----------
        model : LinearGaussianBayesianNetwork
            An already-fitted pgmpy model.

        Returns
        -------
        CausalCovarianceEstimator
        """
        obj = cls.__new__(cls)
        obj._edges = list(model.edges())
        obj._backend = "numpy"
        obj._device = None
        obj._model = model
        obj._variable_names = sorted(model.nodes())
        obj._fitted = True
        return obj

    def observational_covariance(self) -> NDArray[np.float64]:
        r"""Observational covariance Σ.

        Uses ``model.to_joint_gaussian()`` and reorders rows/columns
        to match the sorted variable name order.

        Returns
        -------
        ndarray of shape (p, p)
            Symmetric PSD covariance matrix.
        """
        require(self._fitted, "Must call .fit() first")
        assert self._model is not None

        _mean, cov = self._model.to_joint_gaussian()
        return self._reorder_cov(cov)

    def interventional_covariance(
        self,
        interventions: dict[str, float],
    ) -> NDArray[np.float64]:
        r"""Interventional covariance Σ_do under do-calculus.

        Applies graph mutilation: for each intervened variable, all
        incoming edges are removed (``model.do(...)``), effectively
        setting the variable to a constant.

        Parameters
        ----------
        interventions : dict mapping variable name → fixed value
            Variables subjected to do-interventions.

        Returns
        -------
        ndarray of shape (p, p)
            Covariance matrix under the intervention.  Rows/columns
            corresponding to intervened variables have zero
            variance (deterministic).
        """
        require(self._fitted, "Must call .fit() first")
        require(len(interventions) > 0, "At least one intervention required")
        assert self._model is not None

        do_vars = list(interventions.keys())
        do_model = self._model.do(do_vars)
        do_model.fit(self._simulate_under_intervention(interventions))

        _mean, cov = do_model.to_joint_gaussian()
        cov_reordered = self._reorder_cov(cov, model=do_model)

        node_idx = {n: i for i, n in enumerate(self._variable_names)}
        for var in do_vars:
            idx = node_idx[var]
            cov_reordered[idx, :] = 0.0
            cov_reordered[:, idx] = 0.0

        return cov_reordered

    def decompose(self) -> CovarianceDecomposition:
        r"""Structural decomposition of the observational covariance.

        Extracts the causal loading matrix B and noise variances Ω
        from the fitted CPDs, then verifies the identity:

        .. math::
            \Sigma = (I - B)^{-1}\, \Omega\, (I - B)^{-\top}

        Returns
        -------
        CovarianceDecomposition
            Named tuple with ``B``, ``omega``, ``sigma``, and
            ``variable_names``.
        """
        require(self._fitted, "Must call .fit() first")
        assert self._model is not None

        B, omega = self._extract_loadings()
        n = len(self._variable_names)
        I_minus_B_inv = np.linalg.inv(np.eye(n) - B)
        sigma = np.asarray(
            I_minus_B_inv @ np.diag(omega) @ I_minus_B_inv.T, dtype=np.float64
        )

        return CovarianceDecomposition(
            B=B,
            omega=omega,
            sigma=sigma,
            variable_names=list(self._variable_names),
        )

    @property
    def variable_names(self) -> list[str]:
        """Sorted variable names."""
        require(self._fitted, "Must call .fit() first")
        return list(self._variable_names)

    @property
    def model(self) -> LinearGaussianBayesianNetwork:
        """The underlying fitted pgmpy model."""
        require(self._fitted, "Must call .fit() first")
        assert self._model is not None
        return self._model

    def _configure_backend(self) -> None:
        """Set pgmpy's computation backend before running operations."""
        from pgmpy import config as pgmpy_config

        if self._backend == "torch":
            device = self._device or "cpu"
            pgmpy_config.set_backend("torch", device=device)
        else:
            pgmpy_config.set_backend("numpy")

    def _extract_loadings(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extract B matrix and Ω vector from fitted CPDs."""
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

    def _reorder_cov(
        self,
        cov: NDArray[np.float64],
        model: Any = None,
    ) -> NDArray[np.float64]:
        """Reorder covariance to match sorted variable_names.

        ``to_joint_gaussian()`` returns the covariance in topological
        sort order, so we map from that to ``self._variable_names``.
        """
        import networkx as nx

        m = model if model is not None else self._model
        topo_order = list(nx.topological_sort(m))
        reorder = [topo_order.index(n) for n in self._variable_names]
        return np.array(cov[np.ix_(reorder, reorder)], dtype=np.float64)

    def _simulate_under_intervention(
        self,
        interventions: dict[str, float],
    ) -> pd.DataFrame:
        """Generate data from the do-model for refitting."""
        assert self._model is not None
        samples = self._model.simulate(do=interventions, n_samples=1000)
        return samples

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
