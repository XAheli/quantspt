"""Framework-agnostic model wrappers for quantspt.

These wrappers adapt models from ANY framework into the
``GeneratingFunctionModel`` protocol so they integrate with the
core SPT framework (master formula, drift, Fernholz weights, etc.).

Supported patterns:

    wrap_torch_model(model)     — Any PyTorch nn.Module
    wrap_callable(func)         — Any Python callable G: ℝⁿ → ℝ
    wrap_sklearn_estimator(est) — Any sklearn-style estimator

Each wrapper validates SPT invariants (positivity, concavity)
and provides autograd-based or finite-difference derivatives.

Examples
--------
>>> import torch.nn as nn
>>> from quantspt.ml.wrappers import wrap_torch_model
>>>
>>> class MyNet(nn.Module):
...     def forward(self, x):
...         return -(x**2).sum(dim=-1)  # concave function
>>>
>>> gf = wrap_torch_model(MyNet(), n_assets=5)
>>> weights = gf.to_generating_function().weights(current_mu)

References
----------
Monoyios & Pricilia, arXiv:2506.19715, June 2025.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..core.generating_functions import GeneratingFunction, fernholz_weights
from ..errors import SPTInvariantError
from ._protocols import LearnedGeneratingFunction

if TYPE_CHECKING:
    from collections.abc import Callable


class TorchModelWrapper:
    """Wraps any PyTorch nn.Module as a GeneratingFunctionModel.

    The module must compute a **convex** function f(x) such that
    G_θ(x) = −f(x) + offset is concave and positive. Alternatively,
    if ``negate=False``, the module directly outputs G(x) and the user
    is responsible for ensuring concavity.

    Parameters
    ----------
    model : nn.Module
        PyTorch module. forward(x) takes shape (..., n) → (...,).
    n_assets : int
        Number of input features (assets).
    positivity_offset : float
        Constant added: G = -f + offset. Default 1.0.
    negate : bool
        If True (default), G = -model(x) + offset (model outputs convex f).
        If False, G = model(x) (model directly outputs concave G).
    device : str
        PyTorch device. Default 'cpu'.
    validate : bool
        If True, run concavity/positivity checks on construction.

    Examples
    --------
    >>> model = MyICNN(n_inputs=10)
    >>> wrapper = TorchModelWrapper(model, n_assets=10)
    >>> G_val = wrapper.generating_function(mu)
    >>> grad = wrapper.log_gradient(mu)
    """

    def __init__(
        self,
        model: Any,
        n_assets: int,
        positivity_offset: float = 1.0,
        negate: bool = True,
        device: str = "cpu",
        validate: bool = True,
    ) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TorchModelWrapper requires PyTorch. "
                "Install with: pip install quantspt[ml]"
            ) from e

        self._model = model
        self._n_assets = n_assets
        self._offset = positivity_offset
        self._negate = negate
        self._device = device
        self._model.to(device)
        self._model.eval()

        if validate:
            self._validate()

    def fit(
        self,
        market_weights: NDArray[np.float64],
        *,
        returns: NDArray[np.float64] | None = None,
        validation_split: float = 0.1,
        **kwargs: Any,
    ) -> TorchModelWrapper:
        """No-op for pre-trained models. Override for trainable models.

        The wrapper is usable immediately after construction (the model is
        assumed already trained). This method exists for protocol compliance.
        """
        return self

    def _validate(self) -> None:
        """Spot-check positivity and concavity on random simplex points."""

        rng = np.random.default_rng(42)
        for _ in range(3):
            alpha = rng.exponential(size=self._n_assets)
            mu = alpha / alpha.sum()
            G_val = self.generating_function(mu)
            if G_val <= 0:
                raise SPTInvariantError(
                    f"G(μ) must be positive, got {G_val:.6f} at a test point. "
                    f"Increase positivity_offset or check your model."
                )

    def _G_value_torch(self, x: Any) -> Any:
        """Compute G_θ(x) as a torch tensor (differentiable)."""
        if self._negate:
            return -self._model(x) + self._offset
        return self._model(x)

    def generating_function(self, mu: NDArray[np.float64]) -> float:
        """Evaluate G_θ(μ)."""
        import torch

        mu_t = torch.tensor(mu, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            G_val = self._G_value_torch(mu_t.unsqueeze(0)).squeeze()
        return float(G_val.item())

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute ∇ log G_θ(μ) via PyTorch autograd."""
        import torch

        mu_t = torch.tensor(
            mu, dtype=torch.float32, device=self._device
        ).requires_grad_(True)

        G_val = self._G_value_torch(mu_t.unsqueeze(0)).squeeze()
        G_val = torch.clamp(G_val, min=1e-8)
        log_G = torch.log(G_val)

        (grad,) = torch.autograd.grad(log_G, mu_t)
        return grad.detach().cpu().numpy().astype(np.float64)

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute D²G_θ(μ) via torch.autograd.functional.hessian."""
        import torch

        mu_t = torch.tensor(mu, dtype=torch.float64, device=self._device)

        def G_func(x: torch.Tensor) -> torch.Tensor:
            x32 = x.float()
            if self._negate:
                return (
                    -self._model(x32.unsqueeze(0)).squeeze() + self._offset
                ).double()
            return self._model(x32.unsqueeze(0)).squeeze().double()

        H = torch.autograd.functional.hessian(G_func, mu_t)
        H_np = H.detach().cpu().numpy().astype(np.float64)  # type: ignore[union-attr]
        return (H_np + H_np.T) / 2.0

    def weights(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute portfolio weights via Fernholz formula."""
        grad_log_G = self.log_gradient(mu)
        pi = fernholz_weights(grad_log_G, mu)
        pi = np.maximum(pi, 0.0)
        s = pi.sum()
        if s > 0:
            pi /= s
        return pi

    def to_generating_function(self) -> GeneratingFunction:
        """Convert to core GeneratingFunction ABC."""
        return LearnedGeneratingFunction(
            self,
            name_str="TorchWrapped",
            n_assets=self._n_assets,
            skip_validation=True,
        )


class CallableWrapper:
    """Wraps any Python callable G: ℝⁿ → ℝ as a GeneratingFunctionModel.

    Uses central finite differences for gradient and Hessian computation.
    For production use, prefer autograd-based wrappers (PyTorch/JAX).

    Parameters
    ----------
    func : Callable[[NDArray], float]
        The generating function G(μ) → ℝ₊. Must be positive and
        (ideally) concave on the simplex.
    n_assets : int | None
        Number of assets. If None, inferred on first call.
    h : float
        Step size for finite differences. Default 1e-7.
    validate : bool
        If True, check positivity on construction.

    Examples
    --------
    >>> def entropy_g(mu):
    ...     return np.exp(-np.sum(mu * np.log(mu)))
    >>> gf = CallableWrapper(entropy_g)
    >>> weights = gf.to_generating_function().weights(mu)
    """

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        n_assets: int | None = None,
        h: float = 1e-7,
        validate: bool = True,
    ) -> None:
        self._func = func
        self._n_assets = n_assets
        self._h = h
        if validate and n_assets is not None:
            self._validate()

    def fit(
        self,
        market_weights: NDArray[np.float64],
        *,
        returns: NDArray[np.float64] | None = None,
        validation_split: float = 0.1,
        **kwargs: Any,
    ) -> CallableWrapper:
        """No-op — callable wrappers are ready at construction.

        Exists for protocol compliance.
        """
        if self._n_assets is None:
            self._n_assets = market_weights.shape[1]
        return self

    def _validate(self) -> None:
        """Check positivity on random simplex point."""
        assert self._n_assets is not None
        rng = np.random.default_rng(42)
        alpha = rng.exponential(size=self._n_assets)
        mu = alpha / alpha.sum()
        val = self._func(mu)
        if val <= 0:
            raise SPTInvariantError(
                f"G(μ) must be positive, got {val:.6f}. "
                f"Ensure your function is positive on the simplex."
            )

    def generating_function(self, mu: NDArray[np.float64]) -> float:
        """Evaluate G(μ)."""
        return float(self._func(mu))

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Central-difference ∇ log G(μ)."""
        n = len(mu)
        grad = np.zeros(n)
        h = self._h
        for k in range(n):
            mu_p = mu.copy()
            mu_p[k] += h
            mu_m = mu.copy()
            mu_m[k] -= h
            grad[k] = (
                np.log(max(self._func(mu_p), 1e-30))
                - np.log(max(self._func(mu_m), 1e-30))
            ) / (2 * h)
        return grad

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Finite-difference Hessian D²G(μ)."""
        n = len(mu)
        H = np.zeros((n, n))
        h = self._h
        G0 = self._func(mu)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    mu_p = mu.copy()
                    mu_p[i] += h
                    mu_m = mu.copy()
                    mu_m[i] -= h
                    H[i, i] = (self._func(mu_p) - 2 * G0 + self._func(mu_m)) / h**2
                else:
                    mu_pp = mu.copy()
                    mu_pp[i] += h
                    mu_pp[j] += h
                    mu_pm = mu.copy()
                    mu_pm[i] += h
                    mu_pm[j] -= h
                    mu_mp = mu.copy()
                    mu_mp[i] -= h
                    mu_mp[j] += h
                    mu_mm = mu.copy()
                    mu_mm[i] -= h
                    mu_mm[j] -= h
                    H[i, j] = (
                        self._func(mu_pp)
                        - self._func(mu_pm)
                        - self._func(mu_mp)
                        + self._func(mu_mm)
                    ) / (4 * h**2)
                    H[j, i] = H[i, j]
        return H

    def weights(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Portfolio weights via Fernholz formula."""
        grad_log_G = self.log_gradient(mu)
        pi = fernholz_weights(grad_log_G, mu)
        pi = np.maximum(pi, 0.0)
        s = pi.sum()
        if s > 0:
            pi /= s
        return pi

    def to_generating_function(self) -> GeneratingFunction:
        """Convert to core GeneratingFunction ABC."""
        n = self._n_assets or 5
        return LearnedGeneratingFunction(
            self,
            name_str="CallableWrapped",
            n_assets=n,
            skip_validation=True,
        )


class SklearnWrapper:
    """Wraps a scikit-learn estimator for generating function learning.

    The estimator is trained to predict the generating function value
    G(μ) from market weight inputs. After fitting, it provides the
    GeneratingFunctionModel interface with finite-difference derivatives.

    Parameters
    ----------
    estimator : object
        Any sklearn estimator with fit(X, y) and predict(X) methods.
    n_assets : int | None
        Number of assets. Inferred from data if None.
    h : float
        Step size for finite differences.
    target : str
        What the estimator learns:
        - 'generating_function': learns G(μ) directly
        - 'log_generating_function': learns log G(μ)

    Examples
    --------
    >>> from sklearn.kernel_ridge import KernelRidge
    >>> est = KernelRidge(kernel='rbf')
    >>> wrapper = SklearnWrapper(est, target='generating_function')
    >>> wrapper.fit(market_weights, g_values=target_g)
    """

    def __init__(
        self,
        estimator: Any,
        n_assets: int | None = None,
        h: float = 1e-5,
        target: str = "generating_function",
    ) -> None:
        if target not in ("generating_function", "log_generating_function"):
            raise ValueError(
                f"target must be 'generating_function' or "
                f"'log_generating_function', got {target!r}"
            )
        self._estimator = estimator
        self._n_assets = n_assets
        self._h = h
        self._target = target
        self._fitted = False

    def fit(
        self,
        market_weights: NDArray[np.float64],
        *,
        g_values: NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> SklearnWrapper:
        """Fit the sklearn estimator.

        Parameters
        ----------
        market_weights : ndarray of shape (T, n)
            Market weight vectors as training inputs.
        g_values : ndarray of shape (T,)
            Target values for G(μ) or log G(μ). If None, a default
            target (diversity-weighted G) is computed from the weights.
        **kwargs
            Passed to estimator.fit().

        Returns
        -------
        SklearnWrapper
            The fitted wrapper.
        """
        _T, n = market_weights.shape
        self._n_assets = n

        if g_values is None:
            g_values = np.sum(market_weights**0.5, axis=1)
            if self._target == "log_generating_function":
                g_values = np.log(np.maximum(g_values, 1e-30))

        self._estimator.fit(market_weights, g_values, **kwargs)
        self._fitted = True
        return self

    def generating_function(self, mu: NDArray[np.float64]) -> float:
        """Evaluate G(μ) via the sklearn estimator."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")
        pred = float(self._estimator.predict(mu.reshape(1, -1))[0])
        if self._target == "log_generating_function":
            return float(np.exp(pred))
        return max(pred, 1e-10)

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Central-difference ∇ log G(μ)."""
        n = len(mu)
        grad = np.zeros(n)
        h = self._h
        for k in range(n):
            mu_p = mu.copy()
            mu_p[k] += h
            mu_m = mu.copy()
            mu_m[k] -= h
            g_p = max(self.generating_function(mu_p), 1e-30)
            g_m = max(self.generating_function(mu_m), 1e-30)
            grad[k] = (np.log(g_p) - np.log(g_m)) / (2 * h)
        return grad

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Finite-difference Hessian D²G(μ)."""
        n = len(mu)
        H = np.zeros((n, n))
        h = self._h
        G0 = self.generating_function(mu)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    mu_p = mu.copy()
                    mu_p[i] += h
                    mu_m = mu.copy()
                    mu_m[i] -= h
                    H[i, i] = (
                        self.generating_function(mu_p)
                        - 2 * G0
                        + self.generating_function(mu_m)
                    ) / h**2
                else:
                    mu_pp = mu.copy()
                    mu_pp[i] += h
                    mu_pp[j] += h
                    mu_pm = mu.copy()
                    mu_pm[i] += h
                    mu_pm[j] -= h
                    mu_mp = mu.copy()
                    mu_mp[i] -= h
                    mu_mp[j] += h
                    mu_mm = mu.copy()
                    mu_mm[i] -= h
                    mu_mm[j] -= h
                    H[i, j] = (
                        self.generating_function(mu_pp)
                        - self.generating_function(mu_pm)
                        - self.generating_function(mu_mp)
                        + self.generating_function(mu_mm)
                    ) / (4 * h**2)
                    H[j, i] = H[i, j]
        return H

    def weights(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """Portfolio weights via Fernholz formula."""
        grad_log_G = self.log_gradient(mu)
        pi = fernholz_weights(grad_log_G, mu)
        pi = np.maximum(pi, 0.0)
        s = pi.sum()
        if s > 0:
            pi /= s
        return pi

    def to_generating_function(self) -> GeneratingFunction:
        """Convert to core GeneratingFunction ABC."""
        n = self._n_assets or 5
        return LearnedGeneratingFunction(
            self,
            name_str="SklearnWrapped",
            n_assets=n,
            skip_validation=True,
        )


class JaxFunctionWrapper:
    """Wraps any pure JAX function as a GeneratingFunctionModel.

    Accepts any function fn: jnp.ndarray(n,) → scalar. Pre-JITs gradient
    and Hessian at wrap time for efficient repeated evaluation.

    Parameters
    ----------
    func : callable
        JAX-compatible function G(μ) → scalar. Can close over Flax params,
        Equinox models, or any JAX pytree.
    n_assets : int
        Number of assets.
    validate : bool
        Check positivity on construction.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> def my_G(mu): return -jnp.sum(mu**2) + 2.0
    >>> wrapper = JaxFunctionWrapper(my_G, n_assets=5)
    >>>
    >>> # Flax model:
    >>> fn = lambda mu: model.apply(params, mu)
    >>> wrapper = JaxFunctionWrapper(fn, n_assets=5)
    """

    def __init__(
        self,
        func: Any,
        n_assets: int,
        validate: bool = True,
    ) -> None:
        try:
            import jax
        except ImportError as e:
            raise ImportError(
                "JaxFunctionWrapper requires JAX. "
                "Install with: pip install quantspt[gpu]"
            ) from e

        import jax
        import jax.numpy as jnp

        self._func = func
        self._n_assets = n_assets

        self._jit_G = jax.jit(func)
        self._jit_grad_log_G = jax.jit(jax.grad(lambda mu: jnp.log(func(mu))))
        self._jit_hessian_G = jax.jit(jax.hessian(func))

        if validate:
            self._validate()

    def _validate(self) -> None:
        import jax.numpy as jnp

        rng_key = np.random.default_rng(42)
        alpha = rng_key.exponential(size=self._n_assets)
        mu = alpha / alpha.sum()
        mu_jax = jnp.array(mu)
        val = float(self._jit_G(mu_jax))
        if val <= 0:
            raise SPTInvariantError(
                f"G(μ) must be positive, got {val:.6f}. Check your JAX function."
            )

    def fit(
        self,
        market_weights: NDArray[np.float64],
        *,
        returns: NDArray[np.float64] | None = None,
        validation_split: float = 0.1,
        **kwargs: Any,
    ) -> JaxFunctionWrapper:
        """No-op — JAX wrappers are ready at construction."""
        return self

    def generating_function(self, mu: NDArray[np.float64]) -> float:
        import jax.numpy as jnp

        return float(self._jit_G(jnp.array(mu)))

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        import jax.numpy as jnp

        grad = self._jit_grad_log_G(jnp.array(mu))
        return np.asarray(grad, dtype=np.float64)

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        import jax.numpy as jnp

        H = self._jit_hessian_G(jnp.array(mu))
        H_np = np.asarray(H, dtype=np.float64)
        return (H_np + H_np.T) / 2.0

    def weights(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        grad_log_G = self.log_gradient(mu)
        pi = fernholz_weights(grad_log_G, mu)
        pi = np.maximum(pi, 0.0)
        s = pi.sum()
        if s > 0:
            pi /= s
        return pi

    def to_generating_function(self) -> GeneratingFunction:
        return LearnedGeneratingFunction(
            self, name_str="JaxWrapped", n_assets=self._n_assets, skip_validation=True
        )


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def wrap_torch_model(
    model: Any,
    n_assets: int,
    *,
    positivity_offset: float = 1.0,
    negate: bool = True,
    device: str = "cpu",
    validate: bool = True,
) -> TorchModelWrapper:
    """Wrap any PyTorch nn.Module as a GeneratingFunctionModel.

    Parameters
    ----------
    model : nn.Module
        PyTorch module. forward(x) takes (..., n) → (...,).
        If ``negate=True``, model outputs f(x) (convex) and G = -f + offset.
        If ``negate=False``, model directly outputs G(x) (must be concave).
    n_assets : int
        Number of assets.
    positivity_offset : float
        Constant ensuring G > 0. Only used if negate=True.
    negate : bool
        Whether to negate the model output. Default True.
    device : str
        PyTorch device.
    validate : bool
        Run positivity/concavity spot-checks.

    Returns
    -------
    TorchModelWrapper
        Wrapper implementing GeneratingFunctionModel.
    """
    return TorchModelWrapper(
        model=model,
        n_assets=n_assets,
        positivity_offset=positivity_offset,
        negate=negate,
        device=device,
        validate=validate,
    )


def wrap_callable(
    func: Callable[[NDArray[np.float64]], float],
    n_assets: int | None = None,
    *,
    h: float = 1e-7,
    validate: bool = True,
) -> CallableWrapper:
    """Wrap any Python callable G(μ) → float as a GeneratingFunctionModel.

    Uses finite differences for derivatives. For better precision,
    use ``wrap_torch_model`` with autograd.

    Parameters
    ----------
    func : callable
        G(μ) → float. Must be positive (and ideally concave) on Δ_n⁺.
    n_assets : int, optional
        Number of assets. If None, inferred on first evaluation.
    h : float
        Finite difference step size.
    validate : bool
        Check positivity on construction (requires n_assets).

    Returns
    -------
    CallableWrapper
        Wrapper implementing GeneratingFunctionModel.
    """
    return CallableWrapper(func=func, n_assets=n_assets, h=h, validate=validate)


def wrap_jax_function(
    func: Any,
    n_assets: int,
    *,
    validate: bool = True,
) -> JaxFunctionWrapper:
    """Wrap any pure JAX function as a GeneratingFunctionModel.

    Pre-JITs grad(log G) and hessian(G) at wrap time for fast evaluation.
    Supports Flax, Equinox, or raw JAX functions.

    Parameters
    ----------
    func : callable
        JAX function G(μ) → scalar. May close over params/pytrees.
    n_assets : int
        Number of assets.
    validate : bool
        Run positivity check on construction.

    Returns
    -------
    JaxFunctionWrapper
    """
    return JaxFunctionWrapper(func=func, n_assets=n_assets, validate=validate)


def wrap_sklearn_estimator(
    estimator: Any,
    *,
    n_assets: int | None = None,
    target: str = "generating_function",
    h: float = 1e-5,
) -> SklearnWrapper:
    """Wrap any sklearn estimator as a GeneratingFunctionModel.

    The estimator is trained to predict G(μ) from market weight inputs.
    After fitting, finite differences provide derivatives.

    Parameters
    ----------
    estimator : object
        Any sklearn estimator with fit(X, y) and predict(X).
    n_assets : int, optional
        Number of assets.
    target : str
        'generating_function' or 'log_generating_function'.
    h : float
        Finite difference step size.

    Returns
    -------
    SklearnWrapper
        Wrapper implementing GeneratingFunctionModel.
    """
    return SklearnWrapper(estimator=estimator, n_assets=n_assets, h=h, target=target)


__all__ = [
    "CallableWrapper",
    "JaxFunctionWrapper",
    "SklearnWrapper",
    "TorchModelWrapper",
    "wrap_callable",
    "wrap_jax_function",
    "wrap_sklearn_estimator",
    "wrap_torch_model",
]
