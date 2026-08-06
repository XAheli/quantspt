"""Composable loss functions for Neural FGP training.

Loss functions are first-class objects that can be combined, weighted,
and passed to any training loop. Users can write their own.

Usage::

    from quantspt.ml.losses import relative_return_loss, turnover_penalty

    # Combine losses with arithmetic
    loss_fn = relative_return_loss + 0.1 * turnover_penalty

    # Use in training
    model.fit(data, loss_fn=loss_fn)

    # Or in a custom loop
    for batch in dataloader:
        loss = loss_fn(model, batch)
        loss.backward()

References
----------
Monoyios & Pricilia, arXiv:2506.19715, Eq. 3.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FGPLoss(Protocol):
    """Protocol for FGP training loss functions.

    Any callable matching this signature can serve as a loss.
    The loss receives portfolio weights (computed from the model)
    and returns data for relative wealth computation.
    """

    def __call__(
        self,
        weights_sequence: Any,
        returns_sequence: Any,
        **kwargs: Any,
    ) -> Any:
        """Compute loss given portfolio weights and period returns.

        Parameters
        ----------
        weights_sequence : Tensor of shape (T, n)
            Portfolio weights at each time step.
        returns_sequence : Tensor of shape (T, n)
            Per-asset returns x_{t,i}/x_{t-1,i}.
        **kwargs
            Extra context (e.g., previous weights for turnover).

        Returns
        -------
        Tensor (scalar)
            Loss value to minimize.
        """
        ...


@dataclass(frozen=True)
class _CompositeLoss:
    """Arithmetic combination of losses."""

    terms: tuple[tuple[float, Any], ...]

    def __call__(
        self,
        weights_sequence: Any,
        returns_sequence: Any,
        **kwargs: Any,
    ) -> Any:
        total = None
        for coeff, loss_fn in self.terms:
            val = loss_fn(weights_sequence, returns_sequence, **kwargs)
            term = coeff * val
            total = term if total is None else total + term
        return total

    def __add__(self, other: Any) -> _CompositeLoss:
        if isinstance(other, _CompositeLoss):
            return _CompositeLoss(terms=self.terms + other.terms)
        return _CompositeLoss(terms=(*self.terms, (1.0, other)))

    def __radd__(self, other: Any) -> _CompositeLoss:
        if isinstance(other, _CompositeLoss):
            return _CompositeLoss(terms=other.terms + self.terms)
        return _CompositeLoss(terms=((1.0, other), *self.terms))

    def __mul__(self, scalar: float) -> _CompositeLoss:
        return _CompositeLoss(
            terms=tuple((coeff * scalar, fn) for coeff, fn in self.terms)
        )

    def __rmul__(self, scalar: float) -> _CompositeLoss:
        return self.__mul__(scalar)


class _BaseLoss:
    """Base class providing arithmetic composition for losses."""

    def __add__(self, other: Any) -> _CompositeLoss:
        if isinstance(other, _CompositeLoss):
            return _CompositeLoss(terms=((1.0, self), *other.terms))
        return _CompositeLoss(terms=((1.0, self), (1.0, other)))

    def __radd__(self, other: Any) -> _CompositeLoss:
        if isinstance(other, _CompositeLoss):
            return _CompositeLoss(terms=(*other.terms, (1.0, self)))
        return _CompositeLoss(terms=((1.0, other), (1.0, self)))

    def __mul__(self, scalar: float) -> _CompositeLoss:
        return _CompositeLoss(terms=((scalar, self),))

    def __rmul__(self, scalar: float) -> _CompositeLoss:
        return _CompositeLoss(terms=((scalar, self),))


class RelativeReturnLoss(_BaseLoss):
    r"""Maximize log-relative return (arXiv:2506.19715, Eq. 3.3).

    .. math::
        \mathcal{L} = -\frac{1}{T} \log V_T

    where V_T = Π_{s=1}^T Σ_i π_i(x_{s-1}) · (x_{s,i} / x_{s-1,i})
    """

    def __call__(
        self,
        weights_sequence: Any,
        returns_sequence: Any,
        **kwargs: Any,
    ) -> Any:
        import torch

        T = weights_sequence.shape[0]
        period_returns = (weights_sequence * returns_sequence).sum(dim=-1)
        period_returns = torch.clamp(period_returns, min=1e-8)
        log_V = torch.log(period_returns).sum()
        return -(1.0 / max(T, 1)) * log_V


class WeightRegularization(_BaseLoss):
    r"""ℓ₂ penalty on portfolio weight concentration.

    .. math::
        \mathcal{L}_{reg} = \frac{1}{T} \sum_t \|\pi_t\|^2
    """

    def __call__(
        self,
        weights_sequence: Any,
        returns_sequence: Any,
        **kwargs: Any,
    ) -> Any:
        T = weights_sequence.shape[0]
        return (weights_sequence**2).sum() / max(T, 1)


class TurnoverPenalty(_BaseLoss):
    r"""Penalize high portfolio turnover between rebalancing periods.

    .. math::
        \mathcal{L}_{to} = \frac{1}{T-1} \sum_{t=1}^{T-1}
                           \|\pi_t - \pi_{t-1}\|_1
    """

    def __call__(
        self,
        weights_sequence: Any,
        returns_sequence: Any,
        **kwargs: Any,
    ) -> Any:
        import torch

        T = weights_sequence.shape[0]
        if T < 2:
            return torch.tensor(0.0, device=weights_sequence.device)
        diffs = weights_sequence[1:] - weights_sequence[:-1]
        return diffs.abs().sum() / max(T - 1, 1)


class SharpeRelativeLoss(_BaseLoss):
    """Negative Sharpe ratio of per-period relative returns."""

    def __call__(
        self,
        weights_sequence: Any,
        returns_sequence: Any,
        **kwargs: Any,
    ) -> Any:
        period_returns = (weights_sequence * returns_sequence).sum(dim=-1)
        excess = period_returns - 1.0
        mean_r = excess.mean()
        std_r = excess.std() + 1e-8
        return -mean_r / std_r


relative_return_loss = RelativeReturnLoss()
"""Singleton: maximize log-relative return (arXiv:2506.19715, Eq. 3.3)."""

weight_regularization = WeightRegularization()
"""Singleton: ℓ₂ penalty on portfolio weight concentration."""

turnover_penalty = TurnoverPenalty()
"""Singleton: penalize high portfolio turnover."""

sharpe_of_relative_loss = SharpeRelativeLoss()
"""Singleton: negative Sharpe ratio of relative returns."""


def default_loss(weight_decay: float = 1e-4) -> _CompositeLoss:
    """The standard loss from arXiv:2506.19715, Eq. 3.3.

    L(θ) = −(1/T) log(V_T) + λ‖π‖²
    """
    return relative_return_loss + weight_decay * weight_regularization


__all__ = [
    "FGPLoss",
    "RelativeReturnLoss",
    "SharpeRelativeLoss",
    "TurnoverPenalty",
    "WeightRegularization",
    "default_loss",
    "relative_return_loss",
    "sharpe_of_relative_loss",
    "turnover_penalty",
    "weight_regularization",
]
