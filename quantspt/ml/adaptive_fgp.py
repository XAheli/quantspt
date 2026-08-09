"""Adaptive Functionally Generated Portfolios — anchored neural corrections.

Instead of learning the generating function G from scratch via ICNN
(which collapses on real high-dimensional data due to the positivity
offset instability and softplus training plateaus), this module learns
a multiplicative *correction* to a known-good classical generator:

    G_θ(μ) = G_base(μ) · exp(h_θ(μ))

where G_base is a classical generator (e.g. DiversityGenerator(p=0.5))
and h_θ is a small unconstrained neural network initialized near zero.

Key properties:
  - **Automatically positive**: G_base > 0 and exp(·) > 0, so G_θ > 0
    always — no fragile positivity offset required.
  - **Can't collapse**: even when h_θ ≈ 0 (no learning), G_θ ≈ G_base
    recovers the classical solution, which already works well.
  - **No ICNN needed**: h_θ is unconstrained.  Concavity of G_θ is not
    architecturally enforced; instead, the anchor regularisation keeps
    G_θ close to the (concave) G_base, and empirical performance is
    the arbiter.
  - **Low-dimensional effective learning**: h_θ only needs to learn
    small corrections, vastly reducing sample complexity vs learning G
    from scratch on a 50+-dimensional simplex.
  - **Theoretical soundness**: any C² positive G defines a valid FGP
    via the Fernholz weight formula.  Concavity guarantees a non-negative
    drift process (sufficient for outperformance), but is not required
    for the FGP itself to be well-defined.

The weight formula decomposes cleanly:

    ∇ log G_θ = ∇ log G_base + ∇ h_θ

    π_i = (D_i log G_base + D_i h_θ + 1 - Σ μ_k (D_k log G_base + D_k h_θ)) μ_i
        = π_base_i + correction_i(θ)

Training loss (Eq. 3.3 + anchor penalty):

    L(θ) = -(1/T) log V_T + λ_w ‖π‖² + λ_h E[h_θ(μ)²]

The λ_h term penalises large corrections, keeping the model anchored
near the classical solution.  It deviates only when data provides
strong evidence that doing so improves returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.generating_functions import (
    DiversityGenerator,
    GeneratingFunction,
    fernholz_weights,
)
from ._protocols import LearnedGeneratingFunction

_log = logging.getLogger(__name__)


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "AdaptiveFGP requires PyTorch.  Install with: pip install quantspt[ml]"
        ) from e


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveFGPConfig:
    """Configuration for Adaptive FGP training.

    Parameters
    ----------
    base_p : float
        Diversity exponent for the classical anchor generator.
    correction_dims : list of int
        Hidden-layer widths for the correction network h_θ.
    learning_rate : float
        Optimizer learning rate.
    epochs : int
        Maximum training epochs.
    train_window : int
        Trading days per walk-forward training window.
    eval_window : int
        Trading days per walk-forward evaluation window.
    weight_decay : float
        ℓ₂ penalty on portfolio weights (λ_w in loss).
    anchor_strength : float
        ℓ₂ penalty on correction output h_θ (λ_h in loss).
        Higher values keep the model closer to the classical anchor.
    early_stopping_patience : int
        Epochs without validation improvement before stopping.
    min_epochs : int
        Minimum epochs before early stopping can trigger.
    gradient_clip_norm : float
        Maximum gradient norm.
    device : str
        PyTorch device.
    seed : int or None
        Random seed.
    """

    base_p: float = 0.5
    correction_dims: list[int] = field(default_factory=lambda: [32, 16])
    learning_rate: float = 1e-4
    epochs: int = 300
    train_window: int = 200
    eval_window: int = 20
    weight_decay: float = 1e-6
    anchor_strength: float = 0.01
    early_stopping_patience: int = 30
    min_epochs: int = 50
    gradient_clip_norm: float = 1.0
    device: str = "cpu"
    seed: int | None = None


# ---------------------------------------------------------------------------
# Correction network — small, unconstrained
# ---------------------------------------------------------------------------


class _CorrectionNetwork:
    """Small feedforward NN that outputs h_θ(μ) ∈ ℝ.

    Uses Softplus activations (smooth) and initialises near zero so
    that exp(h_θ) ≈ 1 at the start of training — meaning G_θ ≈ G_base.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_dims: list[int],
        device: str = "cpu",
    ) -> None:
        _require_torch()
        import torch.nn as nn

        layers: list[nn.Module] = []
        prev = n_inputs
        for hd in hidden_dims:
            lin = nn.Linear(prev, hd)
            nn.init.normal_(lin.weight, std=0.01)
            nn.init.zeros_(lin.bias)
            layers.extend([lin, nn.Softplus()])
            prev = hd
        out = nn.Linear(prev, 1)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)

        self._net = nn.Sequential(*layers).to(device)

    def __call__(self, x: Any) -> Any:
        return self._net(x).squeeze(-1)

    def parameters(self) -> Any:
        return self._net.parameters()

    def train(self) -> None:
        self._net.train()

    def eval(self) -> None:
        self._net.eval()

    def to(self, device: str) -> _CorrectionNetwork:
        self._net.to(device)
        return self

    def double(self) -> _CorrectionNetwork:
        self._net.double()
        return self

    def float(self) -> _CorrectionNetwork:
        self._net.float()
        return self


# ---------------------------------------------------------------------------
# AdaptiveFGP — the main class
# ---------------------------------------------------------------------------


class AdaptiveFGP:
    """Adaptive Functionally Generated Portfolio.

    Learns G_θ(μ) = G_base(μ) · exp(h_θ(μ)) where G_base is a classical
    diversity generator and h_θ is a small neural correction.

    Parameters
    ----------
    n_assets : int
        Number of assets.
    config : AdaptiveFGPConfig, optional
        Training hyperparameters.

    Examples
    --------
    ::

        model = AdaptiveFGP(n_assets=50)
        model.fit(market_weights)

        pi = model.weights(current_mu)
        G_val = model.generating_function(current_mu)
    """

    def __init__(
        self,
        n_assets: int = 50,
        config: AdaptiveFGPConfig | None = None,
    ) -> None:
        _require_torch()
        self._n_assets = n_assets
        self._config = config or AdaptiveFGPConfig()
        self._base = DiversityGenerator(p=self._config.base_p)
        self._correction: _CorrectionNetwork | None = None
        self._fitted = False
        self._training_history: dict[str, list[float]] = {
            "loss": [],
            "val_loss": [],
        }

    @property
    def config(self) -> AdaptiveFGPConfig:
        return self._config

    @property
    def training_history(self) -> dict[str, list[float]]:
        return self._training_history

    @property
    def n_assets(self) -> int:
        return self._n_assets

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        market_weights: NDArray[np.float64],
        *,
        returns: NDArray[np.float64] | None = None,
        validation_split: float = 0.1,
        **kwargs: Any,
    ) -> AdaptiveFGP:
        """Train on market weight time series.

        Parameters
        ----------
        market_weights : ndarray (T, n)
            Daily market capitalization weights.
        returns : ndarray (T-1, n), optional
            Per-asset daily returns x_{t,i}/x_{t-1,i}.
            Computed from market_weights if not provided.
        validation_split : float
            Unused (walk-forward handles validation internally).
        **kwargs
            Reserved for forward compatibility.
        """
        import torch

        T, n = market_weights.shape
        self._n_assets = n
        self._base = DiversityGenerator(p=self._config.base_p)

        cfg = self._config
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

        self._correction = _CorrectionNetwork(
            n_inputs=n,
            hidden_dims=cfg.correction_dims,
            device=cfg.device,
        )

        if returns is None:
            returns = market_weights[1:] / market_weights[:-1]
            market_weights = market_weights[:-1]
            T = T - 1

        mw_t = torch.tensor(market_weights, dtype=torch.float32, device=cfg.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=cfg.device)

        opt = torch.optim.Adam(self._correction.parameters(), lr=cfg.learning_rate)

        tw, ew = cfg.train_window, cfg.eval_window

        _log.info(
            "Training AdaptiveFGP: n=%d, T=%d, base_p=%.2f, "
            "anchor_strength=%.1e, lr=%.1e",
            n,
            T,
            cfg.base_p,
            cfg.anchor_strength,
            cfg.learning_rate,
        )

        for epoch in range(cfg.epochs):
            ep_losses: list[float] = []
            val_losses: list[float] = []
            start = 0

            while start + tw + ew <= T:
                loss = self._training_step(
                    mw_t[start : start + tw],
                    ret_t[start : start + tw],
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._correction.parameters(),
                    cfg.gradient_clip_norm,
                )
                opt.step()
                ep_losses.append(loss.item())

                with torch.no_grad():
                    with torch.enable_grad():
                        vl = self._training_step(
                            mw_t[start + tw : start + tw + ew],
                            ret_t[start + tw : start + tw + ew],
                        )
                    val_losses.append(vl.item())
                start += ew

            if ep_losses:
                self._training_history["loss"].append(float(np.mean(ep_losses)))
                self._training_history["val_loss"].append(float(np.mean(val_losses)))

            if epoch % 50 == 0:
                _log.info(
                    "  epoch %d: train_loss=%.6f, val_loss=%.6f",
                    epoch,
                    self._training_history["loss"][-1],
                    self._training_history["val_loss"][-1],
                )

            if self._should_early_stop():
                _log.info("Early stopping at epoch %d", epoch)
                break

        self._fitted = True
        _log.info(
            "Training complete after %d epochs",
            len(self._training_history["loss"]),
        )
        return self

    def _training_step(self, market_weights: Any, returns: Any) -> Any:
        """Single training step computing anchored FGP loss."""
        import torch

        T_len = market_weights.shape[0] - 1
        if T_len <= 0:
            return torch.tensor(0.0, device=market_weights.device, requires_grad=True)

        weights_seq = []
        h_vals = []
        for s in range(T_len):
            pi, h_val = self._compute_weights_torch(market_weights[s])
            weights_seq.append(pi)
            h_vals.append(h_val)

        weights_tensor = torch.stack(weights_seq)
        returns_tensor = returns[:T_len]
        h_tensor = torch.stack(h_vals)

        period_returns = (weights_tensor * returns_tensor).sum(dim=-1)
        period_returns = torch.clamp(period_returns, min=1e-8)
        log_V = torch.log(period_returns).sum()
        return_loss = -(1.0 / max(T_len, 1)) * log_V

        weight_penalty = self._config.weight_decay * (
            (weights_tensor**2).sum() / max(T_len, 1)
        )

        anchor_penalty = self._config.anchor_strength * ((h_tensor**2).mean())

        return return_loss + weight_penalty + anchor_penalty

    def _compute_weights_torch(self, mu: Any) -> tuple[Any, Any]:
        """Compute Fernholz weights from G_θ = G_base · exp(h_θ).

        Returns (weights, h_val) so h_val can be used in anchor penalty.
        """
        import torch

        corr = self._correction
        assert corr is not None

        with torch.enable_grad():
            mu = mu.detach().requires_grad_(True)

            mu_np = mu.detach().cpu().numpy().astype(np.float64)
            grad_log_G_base = self._base.log_gradient(mu_np)

            grad_base_t = torch.tensor(
                grad_log_G_base,
                dtype=mu.dtype,
                device=mu.device,
            )

            h_val = corr(mu.unsqueeze(0)).squeeze(0)

            (grad_h,) = torch.autograd.grad(h_val, mu, create_graph=True)

            grad_log_G = grad_base_t + grad_h

            S = (mu * grad_log_G).sum()
            pi = (grad_log_G + 1.0 - S) * mu
            pi = torch.clamp(pi, min=0.0)
            s = pi.sum()
            if s > 0:
                pi = pi / s

            return pi, h_val

    def _should_early_stop(self) -> bool:
        patience = self._config.early_stopping_patience
        min_ep = self._config.min_epochs
        vl = self._training_history["val_loss"]
        if len(vl) < max(patience + 1, min_ep):
            return False
        return min(vl[-patience:]) >= min(vl[:-patience])

    # ------------------------------------------------------------------
    # Evaluation (GeneratingFunctionModel protocol)
    # ------------------------------------------------------------------

    def generating_function(self, mu: NDArray[np.float64]) -> float:
        """Evaluate G_θ(μ) = G_base(μ) · exp(h_θ(μ))."""
        import torch

        if not self._fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        G_base = self._base(mu)

        corr = self._correction
        assert corr is not None
        corr.double()
        mu_t = torch.tensor(mu, dtype=torch.float64, device=self._config.device)
        with torch.no_grad():
            h_val = corr(mu_t.unsqueeze(0)).squeeze().item()
        corr.float()

        return max(G_base * np.exp(h_val), 1e-30)

    def log_gradient(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """∇ log G_θ(μ) = ∇ log G_base(μ) + ∇ h_θ(μ)."""
        import torch

        if not self._fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        grad_base = self._base.log_gradient(mu)

        corr = self._correction
        assert corr is not None
        corr.double()
        mu_t = torch.tensor(
            mu,
            dtype=torch.float64,
            device=self._config.device,
        ).requires_grad_(True)
        h_val = corr(mu_t.unsqueeze(0)).squeeze(0)
        (grad_h,) = torch.autograd.grad(h_val, mu_t)
        corr.float()

        return grad_base + grad_h.detach().cpu().numpy()

    def hessian(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """D²G_θ(μ) via autograd."""
        import torch

        if not self._fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        corr = self._correction
        assert corr is not None
        corr.double()
        mu_t = torch.tensor(mu, dtype=torch.float64, device=self._config.device)

        def G_func(x: torch.Tensor) -> torch.Tensor:
            mu_np = x.detach().cpu().numpy()
            G_base_val = self._base(mu_np)
            h_val = corr(x.unsqueeze(0)).squeeze()
            return torch.tensor(G_base_val, dtype=x.dtype, device=x.device) * torch.exp(
                h_val
            )

        H = torch.autograd.functional.hessian(G_func, mu_t)
        corr.float()
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
            self,
            name_str="AdaptiveFGP",
            n_assets=self._n_assets,
        )


__all__ = ["AdaptiveFGP", "AdaptiveFGPConfig"]
