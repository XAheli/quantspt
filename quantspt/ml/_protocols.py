"""ML Protocols for quantspt — framework-agnostic interfaces.

Every integration point defines a ``typing.Protocol`` so users can
implement with any ML framework (PyTorch, JAX, scikit-learn, NumPy)
and plug into quantspt seamlessly.

References
----------
Monoyios & Pricilia, "Neural Functionally Generated Portfolios,"
arXiv:2506.19715, June 2025.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from ..core.generating_functions import GeneratingFunction


@runtime_checkable
class GeneratingFunctionModel(Protocol):
    """Protocol for any model that learns a generating function G: Δ_n → ℝ₊.

    A generating function must satisfy three mathematical requirements:
      1. Positivity: G(μ) > 0 for all μ ∈ Δ_n⁺
      2. Concavity: The Hessian D²G(μ) is negative semi-definite
      3. Smoothness: G is C² (twice continuously differentiable)

    These properties ensure that the Fernholz weight formula (F&K Survey
    Eq. 11.1) and master formula (Eq. 11.2) remain valid for the learned G.

    Any model implementing this protocol can be converted to a
    ``GeneratingFunction`` (the core ABC) via ``to_generating_function()``,
    enabling seamless use with drift computation, master formula
    verification, arbitrage detection, etc.

    References
    ----------
    Monoyios & Pricilia, "Neural Functionally Generated Portfolios,"
    arXiv:2506.19715, June 2025.
    """

    def fit(
        self,
        market_weights: NDArray[np.float64],
        *,
        returns: NDArray[np.float64] | None = None,
        validation_split: float = 0.1,
        **kwargs: Any,
    ) -> Self:
        """Train the model on historical market weight data.

        Parameters
        ----------
        market_weights : ndarray of shape (T, n)
            Time series of market weight vectors μ(t) ∈ Δ_n⁺.
            Each row must be positive and sum to 1.
        returns : ndarray of shape (T, n), optional
            Simple returns x_{t,i} / x_{t-1,i} for each asset.
            Required for relative-return-based training objectives.
        validation_split : float
            Fraction of data reserved for validation.
        **kwargs
            Implementation-specific hyperparameters.

        Returns
        -------
        Self
            The fitted model (for method chaining).
        """
        ...

    def generating_function(self, mu: NDArray[np.float64]) -> float:
        """Evaluate the learned generating function G_θ(μ).

        Parameters
        ----------
        mu : ndarray of shape (n,)
            Market weight vector in the open simplex Δ_n⁺.

        Returns
        -------
        float
            G_θ(μ) > 0. Must be strictly positive.
        """
        ...

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute ∇ log G_θ(μ) = [D_k log G_θ(μ)]_{k=1}^n.

        Used in the Fernholz weight formula:
            π_i = [D_i log G(μ) + 1 − Σ_k μ_k D_k log G(μ)] μ_i

        Parameters
        ----------
        mu : ndarray of shape (n,)
            Market weight vector.

        Returns
        -------
        ndarray of shape (n,)
            Gradient of log G_θ at μ.
        """
        ...

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the Hessian D²G_θ(μ) ∈ ℝ^{n×n}.

        Used in the drift process computation:
            g(t) = −(1/2G) Σ_{i,j} D²_{ij}G · μ_i μ_j τ^μ_{ij}

        Parameters
        ----------
        mu : ndarray of shape (n,)
            Market weight vector.

        Returns
        -------
        ndarray of shape (n, n)
            Hessian matrix. Must be negative semi-definite for concavity.
        """
        ...

    def to_generating_function(self) -> GeneratingFunction:
        """Convert to the core GeneratingFunction ABC.

        Returns a ``GeneratingFunction`` subclass wrapping this model,
        enabling use with ``drift_process()``,
        ``master_formula_decomposition()``, ``fernholz_weights()``,
        and all other core functions.

        Returns
        -------
        GeneratingFunction
            A wrapped instance compatible with the core API.
        """
        ...


@runtime_checkable
class CovarianceEstimator(Protocol):
    """Protocol for learned covariance rate matrix estimators.

    The covariance rate matrix a_{ij}(t) drives excess growth rates and
    portfolio performance in SPT. Better covariance estimates lead to
    better τ^μ → better drift → better arbitrage detection.

    Implementations must produce symmetric positive semi-definite matrices.
    """

    def fit(
        self,
        returns: NDArray[np.float64],
        *,
        timestamps: NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the covariance model to return data.

        Parameters
        ----------
        returns : ndarray of shape (T, n)
            Historical return matrix.
        timestamps : array-like of shape (T,), optional
            Timestamps for irregular spacing.
        **kwargs
            Model-specific parameters.

        Returns
        -------
        Self
            The fitted model.
        """
        ...

    def estimate(
        self,
        t: int | None = None,
    ) -> NDArray[np.float64]:
        """Estimate the covariance rate matrix at time t.

        Parameters
        ----------
        t : int, optional
            Time index. If None, returns unconditional estimate.

        Returns
        -------
        ndarray of shape (n, n)
            Symmetric PSD covariance rate matrix.
        """
        ...

    @property
    def n_assets(self) -> int:
        """Number of assets in the fitted model."""
        ...


@runtime_checkable
class RegimeDetector(Protocol):
    """Protocol for market regime detection models.

    Regime detection serves a critical role in SPT: the diversity conditions
    (FKK Eq. 4.5) required for relative arbitrage do not always hold.
    A regime detector identifies WHEN conditions are favorable for deploying
    FGP strategies.
    """

    def fit(
        self,
        features: NDArray[np.float64],
        *,
        n_regimes: int = 2,
        **kwargs: Any,
    ) -> Self:
        """Fit the regime model to feature data.

        Parameters
        ----------
        features : ndarray of shape (T, d)
            Feature matrix (diversity measures, growth rates, etc.).
        n_regimes : int
            Number of regimes to detect.
        **kwargs
            Model-specific parameters.

        Returns
        -------
        Self
            The fitted model.
        """
        ...

    def predict(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Predict regime labels for new feature data.

        Parameters
        ----------
        features : ndarray of shape (T, d)
            Feature matrix.

        Returns
        -------
        ndarray of shape (T,)
            Integer regime labels in [0, n_regimes).
        """
        ...

    def predict_proba(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict regime probabilities.

        Parameters
        ----------
        features : ndarray of shape (T, d)
            Feature matrix.

        Returns
        -------
        ndarray of shape (T, n_regimes)
            Probability of each regime at each time step.
        """
        ...

    @property
    def n_regimes(self) -> int:
        """Number of regimes in the fitted model."""
        ...

    @property
    def transition_matrix(self) -> NDArray[np.float64]:
        """Regime transition probability matrix.

        Returns
        -------
        ndarray of shape (n_regimes, n_regimes)
            Row-stochastic transition matrix.
        """
        ...


class LearnedGeneratingFunction(GeneratingFunction):
    """Adapter: wraps any GeneratingFunctionModel as a GeneratingFunction.

    This bridges the ML world (protocols) and the core SPT world (ABCs).
    Once a model is trained, wrapping it in this adapter makes it usable
    with drift_process(), master_formula_decomposition(), fernholz_weights(),
    and every other core function.

    Parameters
    ----------
    model : GeneratingFunctionModel
        A fitted model implementing the GeneratingFunctionModel protocol.
    name_str : str
        Display name for logging and visualization.

    Raises
    ------
    SPTInvariantError
        If the model fails basic validation checks (positivity, concavity
        spot-check on a random simplex point).

    References
    ----------
    Monoyios & Pricilia, "Neural Functionally Generated Portfolios,"
    arXiv:2506.19715, June 2025.
    """

    def __init__(
        self,
        model: GeneratingFunctionModel,
        name_str: str = "LearnedG",
        *,
        n_assets: int = 5,
        skip_validation: bool = False,
    ) -> None:
        self._model = model
        self._name = name_str
        self._n_assets = n_assets
        if not skip_validation:
            self._validate()

    def _validate(self) -> None:
        """Spot-check mathematical invariants on a random simplex point."""
        from ..errors import SPTInvariantError

        rng = np.random.default_rng(42)
        alpha = rng.exponential(size=self._n_assets)
        mu = alpha / alpha.sum()

        G_val = self._model.generating_function(mu)
        if G_val <= 0:
            raise SPTInvariantError(f"G(μ) must be positive, got {G_val} at test point")

        H = self._model.hessian(mu)
        eigenvalues = np.linalg.eigvalsh(H)
        if eigenvalues[-1] > 1e-4:
            raise SPTInvariantError(
                f"Hessian must be negative semi-definite (concavity), "
                f"but largest eigenvalue is {eigenvalues[-1]:.4e}"
            )

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, mu: NDArray[np.float64]) -> float:
        return self._model.generating_function(mu)

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._model.log_gradient(mu)

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._model.hessian(mu)
