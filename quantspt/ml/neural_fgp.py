"""Neural Functionally Generated Portfolios via Input Convex Neural Networks.

Implements the Neural FGP framework from Monoyios & Pricilia
(arXiv:2506.19715, June 2025). The generating function G_θ is parameterized
as the negative of an ICNN output, guaranteeing concavity by construction.

Design Principles:
  - NOTHING is hardcoded — optimizer, loss, architecture all pluggable
  - Users can substitute ANY nn.Module for the ICNN
  - Users can use ``fit()`` OR ``training_step()`` in their own loop
  - Users can pass ANY optimizer via string or callable
  - Users can compose ANY loss from ``quantspt.ml.losses``

References
----------
[1] Monoyios & Pricilia, "Neural Functionally Generated Portfolios,"
    arXiv:2506.19715, June 2025.
[2] Amos, Xu & Kolter, "Input Convex Neural Networks," ICML 2017.
[3] Fernholz & Karatzas, "Stochastic Portfolio Theory: An Overview," 2009.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.generating_functions import GeneratingFunction, fernholz_weights
from ._protocols import LearnedGeneratingFunction


def _require_torch() -> None:
    """Raise ImportError with install instructions if torch is missing."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Neural FGP requires PyTorch. Install with: pip install quantspt[ml]"
        ) from e


# ---------------------------------------------------------------------------
# Configuration — every choice is a parameter
# ---------------------------------------------------------------------------


@dataclass
class NeuralFGPConfig:
    """Configuration for Neural FGP training.

    Every hyperparameter is explicit and overridable. Nothing is hardcoded.

    Attributes
    ----------
    hidden_dims : list of int
        ICNN hidden layer dimensions (only used if no custom network).
    activation : str
        'softplus' (recommended) or 'relu'.
    learning_rate : float
        Optimizer learning rate.
    optimizer : str
        Optimizer name: 'adam', 'adamw', 'sgd', 'lbfgs', 'rmsprop'.
    optimizer_kwargs : dict
        Extra kwargs passed to the optimizer constructor.
    epochs : int
        Maximum training epochs.
    train_window : int
        Trading days per training window (arXiv:2506.19715: 200).
    eval_window : int
        Trading days per evaluation window (arXiv:2506.19715: 20).
    weight_decay : float
        ℓ₂ regularization coefficient λ in the loss (Eq. 3.3).
    positivity_offset : float
        Constant added to ensure G_θ > 0.
    early_stopping_patience : int
        Stop if loss doesn't improve for this many epochs.
    device : str
        PyTorch device: 'cpu', 'cuda', 'mps'.
    seed : int | None
        Random seed for reproducibility.
    gradient_clip_norm : float
        Maximum gradient norm for clipping.
    walk_forward : bool
        If True, use walk-forward validation (arXiv:2506.19715 §3.2.1).
    """

    hidden_dims: list[int] = field(default_factory=lambda: [64, 64, 32])
    activation: str = "softplus"
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    epochs: int = 500
    train_window: int = 200
    eval_window: int = 20
    weight_decay: float = 1e-4
    positivity_offset: float = 1.0
    early_stopping_patience: int = 20
    device: str = "cpu"
    seed: int | None = None
    gradient_clip_norm: float = 1.0
    walk_forward: bool = True


def _build_optimizer(params: Any, config: NeuralFGPConfig) -> Any:
    """Build optimizer from config — supports any torch optimizer by name."""
    import torch.optim

    name = config.optimizer.lower()
    kwargs = {"lr": config.learning_rate, **config.optimizer_kwargs}

    registry: dict[str, type] = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "lbfgs": torch.optim.LBFGS,
        "adagrad": torch.optim.Adagrad,
    }

    if name not in registry:
        raise ValueError(
            f"Unknown optimizer {config.optimizer!r}. "
            f"Supported: {list(registry.keys())}. "
            f"Or pass a custom optimizer to fit()."
        )

    return registry[name](params, **kwargs)


# ---------------------------------------------------------------------------
# InputConvexNN — built-in ICNN with proper parametrization
# ---------------------------------------------------------------------------


def _make_non_negative_parametrization() -> Any:
    """Create an nn.Module parametrization enforcing non-negative weights via softplus."""
    import torch.nn as nn
    import torch.nn.functional as F

    class NonNegative(nn.Module):  # type: ignore[misc]
        def forward(self, x: Any) -> Any:
            return F.softplus(x)

    return NonNegative()


class InputConvexNN:
    """Input Convex Neural Network — built-in architecture with guaranteed convexity.

    Implements Amos et al. (2017) as used in arXiv:2506.19715 §3.1.2.
    Uses ``torch.nn.utils.parametrize`` with softplus to enforce non-negative
    weights continuously (no clamping needed).

    Users may substitute ANY nn.Module. This is just a convenient default.

    Parameters
    ----------
    n_inputs : int
        Number of assets.
    hidden_dims : list of int
        Hidden layer widths.
    activation : str
        'softplus' or 'relu'.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_dims: list[int] | None = None,
        activation: str = "softplus",
    ) -> None:
        _require_torch()
        import torch.nn as nn
        import torch.nn.utils.parametrize as parametrize

        if hidden_dims is None:
            hidden_dims = [64, 64, 32]

        self.n_inputs = n_inputs
        self.hidden_dims = hidden_dims
        K = len(hidden_dims)

        if activation == "softplus":
            self._activation = nn.Softplus()
        elif activation == "relu":
            self._activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation!r}")

        self._W0 = nn.Linear(n_inputs, hidden_dims[0])

        self._Wz = nn.ModuleList()
        self._Ux = nn.ModuleList()
        for k in range(1, K):
            wz = nn.Linear(hidden_dims[k - 1], hidden_dims[k], bias=False)
            parametrize.register_parametrization(
                wz, "weight", _make_non_negative_parametrization()
            )
            self._Wz.append(wz)

            ux = nn.Linear(n_inputs, hidden_dims[k])
            self._Ux.append(ux)

        self._w_out = nn.Linear(hidden_dims[-1], 1, bias=False)
        parametrize.register_parametrization(
            self._w_out, "weight", _make_non_negative_parametrization()
        )
        self._u_out = nn.Linear(n_inputs, 1)

        self._module = nn.Module()
        self._module.W0 = self._W0
        self._module.Wz = self._Wz
        self._module.Ux = self._Ux
        self._module.w_out = self._w_out
        self._module.u_out = self._u_out
        self._module.activation = self._activation

    @property
    def module(self) -> Any:
        """Underlying nn.Module."""
        return self._module

    def parameters(self) -> Any:
        """All learnable parameters."""
        return self._module.parameters()

    def train(self) -> None:
        """Training mode."""
        self._module.train()

    def eval(self) -> None:
        """Evaluation mode."""
        self._module.eval()

    def to(self, device: str) -> InputConvexNN:
        """Move to device."""
        self._module.to(device)
        return self

    def double(self) -> InputConvexNN:
        """Cast all parameters to float64 (for evaluation precision)."""
        self._module.double()
        return self

    def float(self) -> InputConvexNN:
        """Cast all parameters to float32 (restore after evaluation)."""
        self._module.float()
        return self

    def forward(self, x: Any) -> Any:
        """Compute f(x) — the convex function. G_θ = −f + offset."""
        import torch.nn.functional as F

        z = self._activation(self._W0(x))
        for Wz_k, Ux_k in zip(self._Wz, self._Ux, strict=False):
            z = self._activation(
                F.linear(z, Wz_k.weight, None) + F.linear(x, Ux_k.weight, Ux_k.bias)
            )
        f = F.linear(z, self._w_out.weight, None).squeeze(-1) + self._u_out(x).squeeze(
            -1
        )
        return f

    def __call__(self, x: Any) -> Any:
        """Forward pass (calls model hooks like any nn.Module)."""
        return self.forward(x)

    def enforce_constraints(self) -> None:
        """No-op: softplus parametrization handles constraints automatically."""


# ---------------------------------------------------------------------------
# NeuralFGP — modular, extensible high-level interface
# ---------------------------------------------------------------------------


class NeuralFGP:
    """Neural Functionally Generated Portfolio — fully pluggable interface.

    Learns G_θ: Δ_n → ℝ₊ using any PyTorch model (default: InputConvexNN).
    Follows sklearn conventions while exposing ``training_step()`` for
    custom loops.

    NOTHING is hardcoded:
      - Architecture: pass ``network=MyNet()``
      - Optimizer: set ``config.optimizer='adamw'`` or pass to ``fit()``
      - Loss: pass ``loss_fn=my_loss`` (from ``quantspt.ml.losses``)
      - Training loop: use ``training_step()`` in your own loop

    Parameters
    ----------
    n_assets : int
        Number of assets.
    config : NeuralFGPConfig, optional
        Fully configurable training parameters.
    network : object, optional
        Any PyTorch model with forward(x) → scalar.
    loss_fn : callable, optional
        Composable loss from ``quantspt.ml.losses``.

    Examples
    --------
    Standard::

        model = NeuralFGP(n_assets=5)
        model.fit(market_weights)

    Custom everything::

        model = NeuralFGP(
            n_assets=5,
            config=NeuralFGPConfig(optimizer='adamw', learning_rate=5e-4),
            network=MyCustomICNN(5),
            loss_fn=relative_return_loss + 0.1 * turnover_penalty,
        )
        model.fit(data)

    Own training loop::

        model = NeuralFGP(n_assets=5)
        model.setup()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for batch in dataloader:
            loss = model.training_step(batch_mw, batch_ret)
            opt.zero_grad(); loss.backward(); opt.step()
            model.enforce_constraints()

    References
    ----------
    [1] Monoyios & Pricilia, arXiv:2506.19715, June 2025.
    """

    def __init__(
        self,
        n_assets: int = 5,
        config: NeuralFGPConfig | None = None,
        network: Any | None = None,
        loss_fn: Any | None = None,
    ) -> None:
        _require_torch()
        self._n_assets = n_assets
        self._config = config or NeuralFGPConfig()
        self._loss_fn = loss_fn
        self._user_network = network
        self._network: Any | None = None
        self._fitted = False
        self._training_history: dict[str, list[float]] = {"loss": [], "val_loss": []}

    @property
    def config(self) -> NeuralFGPConfig:
        return self._config

    @property
    def training_history(self) -> dict[str, list[float]]:
        return self._training_history

    @property
    def n_assets(self) -> int:
        return self._n_assets

    def parameters(self) -> Any:
        """Model parameters — pass to your own optimizer."""
        if self._network is None:
            raise RuntimeError("Call setup() or fit() first.")
        return self._network.parameters()

    def setup(self, market_weights: NDArray[np.float64] | None = None) -> None:
        """Initialize network. Called automatically by fit(), or manually."""
        import torch

        cfg = self._config
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
        if market_weights is not None:
            self._n_assets = market_weights.shape[1]
        if self._user_network is not None:
            self._network = self._user_network
        else:
            self._network = InputConvexNN(
                n_inputs=self._n_assets,
                hidden_dims=cfg.hidden_dims,
                activation=cfg.activation,
            )
        self._network.to(cfg.device)

    def training_step(self, market_weights: Any, returns: Any) -> Any:
        """Single training step — use in your own loop.

        Parameters
        ----------
        market_weights : Tensor (T, n)
        returns : Tensor (T, n)

        Returns
        -------
        Tensor (scalar) — differentiable loss.
        """
        import torch

        if self._network is None:
            raise RuntimeError("Call setup() first.")

        T = market_weights.shape[0] - 1
        if T <= 0:
            return torch.tensor(0.0, device=market_weights.device, requires_grad=True)

        weights_seq = []
        for s in range(T):
            pi = self._compute_weights_torch(market_weights[s])
            weights_seq.append(pi)

        weights_tensor = torch.stack(weights_seq)
        returns_tensor = returns[:T]

        if self._loss_fn is not None:
            return self._loss_fn(weights_tensor, returns_tensor)
        from .losses import default_loss

        return default_loss(self._config.weight_decay)(weights_tensor, returns_tensor)

    def enforce_constraints(self) -> None:
        """Enforce architectural constraints. No-op for softplus parametrization."""
        if self._network is not None and hasattr(self._network, "enforce_constraints"):
            self._network.enforce_constraints()

    def fit(
        self,
        market_weights: NDArray[np.float64],
        *,
        returns: NDArray[np.float64] | None = None,
        validation_split: float = 0.1,
        loss_fn: Any | None = None,
        optimizer: Any | None = None,
        **kwargs: Any,
    ) -> NeuralFGP:
        """Train — standard sklearn-style entry point.

        Parameters
        ----------
        market_weights : ndarray (T, n)
        returns : ndarray (T, n), optional
        validation_split : float
        loss_fn : callable, optional — override loss
        optimizer : torch.optim.Optimizer, optional — override optimizer
        """
        import torch

        if loss_fn is not None:
            self._loss_fn = loss_fn

        T, n = market_weights.shape
        self._n_assets = n
        self.setup(market_weights)

        if returns is None:
            returns = market_weights[1:] / market_weights[:-1]
            market_weights = market_weights[:-1]
            T = T - 1

        cfg = self._config
        mw_t = torch.tensor(market_weights, dtype=torch.float32, device=cfg.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=cfg.device)

        opt = (
            optimizer
            if optimizer is not None
            else _build_optimizer(self.parameters(), cfg)
        )

        if cfg.walk_forward:
            self._train_walk_forward(mw_t, ret_t, T, opt)
        else:
            split_idx = int(T * (1 - validation_split))
            self._train_single(
                mw_t[:split_idx],
                ret_t[:split_idx],
                mw_t[split_idx:],
                ret_t[split_idx:],
                opt,
            )

        self._fitted = True
        return self

    def _train_walk_forward(self, mw: Any, ret: Any, T: int, opt: Any) -> None:
        import torch

        cfg = self._config
        tw, ew = cfg.train_window, cfg.eval_window
        for _epoch in range(cfg.epochs):
            ep_losses, val_losses = [], []
            start = 0
            while start + tw + ew <= T:
                loss = self.training_step(
                    mw[start : start + tw], ret[start : start + tw]
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), cfg.gradient_clip_norm
                )
                opt.step()
                self.enforce_constraints()
                ep_losses.append(loss.item())

                with torch.no_grad():
                    with torch.enable_grad():
                        vl = self.training_step(
                            mw[start + tw : start + tw + ew],
                            ret[start + tw : start + tw + ew],
                        )
                    val_losses.append(vl.item())
                start += ew

            if ep_losses:
                self._training_history["loss"].append(float(np.mean(ep_losses)))
                self._training_history["val_loss"].append(float(np.mean(val_losses)))
            if self._should_early_stop():
                break

    def _train_single(
        self, train_mw: Any, train_ret: Any, val_mw: Any, val_ret: Any, opt: Any
    ) -> None:
        import torch

        cfg = self._config
        for _epoch in range(cfg.epochs):
            loss = self.training_step(train_mw, train_ret)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.gradient_clip_norm)
            opt.step()
            self.enforce_constraints()
            self._training_history["loss"].append(loss.item())

            with torch.no_grad():
                with torch.enable_grad():
                    vl = self.training_step(val_mw, val_ret)
                self._training_history["val_loss"].append(vl.item())
            if self._should_early_stop():
                break

    def _compute_weights_torch(self, mu: Any) -> Any:
        """Fernholz weights via autograd (arXiv:2506.19715 Eq. 3.1)."""
        import torch

        net = self._network
        assert net is not None
        with torch.enable_grad():
            mu = mu.detach().requires_grad_(True)
            f_val = net(mu.unsqueeze(0)).squeeze(0)
            G_val = -f_val + self._config.positivity_offset
            G_val = torch.clamp(G_val, min=1e-8)
            log_G = torch.log(G_val)
            (grad_log_G,) = torch.autograd.grad(log_G, mu, create_graph=True)
            S = (mu * grad_log_G).sum()
            pi = (grad_log_G + 1.0 - S) * mu
            pi = torch.clamp(pi, min=0.0)
            s = pi.sum()
            if s > 0:
                pi = pi / s
            return pi

    def _should_early_stop(self) -> bool:
        patience = self._config.early_stopping_patience
        vl = self._training_history["val_loss"]
        if len(vl) < patience + 1:
            return False
        return min(vl[-patience:]) >= min(vl[:-patience])

    # ------------------------------------------------------------------
    # GeneratingFunctionModel protocol
    # ------------------------------------------------------------------

    def generating_function(self, mu: NDArray[np.float64]) -> float:
        """Evaluate G_θ(μ) = −f(μ) + offset.

        Evaluates in float64 regardless of training dtype because
        downstream weight/drift computations (relative_covariance,
        excess_growth_rate) suffer catastrophic cancellation in float32.
        This is the Category C boundary: training is float32, evaluation
        promotes to float64.
        """
        import torch

        if not self._fitted:
            raise RuntimeError("Model must be fitted before evaluation.")
        net = self._network
        assert net is not None
        net.double()
        mu_t = torch.tensor(
            mu, dtype=torch.float64, device=self._config.device
        ).unsqueeze(0)
        with torch.no_grad():
            f_val = net(mu_t).squeeze().item()
        net.float()
        return max(-f_val + self._config.positivity_offset, 1e-10)

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """∇ log G_θ(μ) via autograd.

        Evaluates in float64: the gradient feeds into Fernholz weight
        computation (π_i = μ_i D_i log G + μ_i), where catastrophic
        cancellation occurs when weights are close to market weights.
        """
        import torch

        if not self._fitted:
            raise RuntimeError("Model must be fitted before evaluation.")
        net = self._network
        assert net is not None
        net.double()
        mu_t = torch.tensor(
            mu, dtype=torch.float64, device=self._config.device
        ).requires_grad_(True)
        f_val = net(mu_t.unsqueeze(0)).squeeze(0)
        G_val = torch.clamp(-f_val + self._config.positivity_offset, min=1e-8)
        log_G = torch.log(G_val)
        (grad,) = torch.autograd.grad(log_G, mu_t)
        net.float()
        return grad.detach().cpu().numpy()

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """D²G_θ(μ) via autograd.

        Float64 required: the Hessian D²G feeds into drift process
        computation (γ* = ½ Σ (π_i - μ_i)(π_j - μ_j) τ_{ij}),
        where second-order differences amplify any precision loss.
        """
        import torch

        if not self._fitted:
            raise RuntimeError("Model must be fitted before evaluation.")
        net = self._network
        assert net is not None
        net.double()
        mu_t = torch.tensor(mu, dtype=torch.float64, device=self._config.device)

        def G_func(x: torch.Tensor) -> torch.Tensor:
            return -net(x.unsqueeze(0)).squeeze() + self._config.positivity_offset

        H = torch.autograd.functional.hessian(G_func, mu_t)
        net.float()
        H_np = H.detach().cpu().numpy()  # type: ignore[union-attr]
        return (H_np + H_np.T) / 2.0

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
        if not self._fitted:
            raise RuntimeError("Model must be fitted before conversion.")
        return LearnedGeneratingFunction(
            self, name_str="NeuralFGP", n_assets=self._n_assets
        )


__all__ = ["InputConvexNN", "NeuralFGP", "NeuralFGPConfig"]
