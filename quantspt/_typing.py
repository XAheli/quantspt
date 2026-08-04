"""Type aliases and protocols used across quantspt.

Typed aliases document intent at the signature level:
``Time`` is not just a float — it is a point on the continuous time axis.
"""

from __future__ import annotations

from typing import NewType, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# -- scalar aliases (not runtime-enforced, but self-documenting) ---------------

Time = NewType("Time", float)
Weight = NewType("Weight", float)
CovarianceRate = NewType("CovarianceRate", float)
DiversityParameter = NewType("DiversityParameter", float)
GrowthRate = NewType("GrowthRate", float)

# -- array aliases ------------------------------------------------------------

WeightVector = NDArray[np.float64]
CovarianceMatrix = NDArray[np.float64]
ReturnSeries = NDArray[np.float64]


# -- protocols ----------------------------------------------------------------


@runtime_checkable
class StochasticProcess(Protocol):
    """Protocol for continuous-time stochastic processes.

    Separates the continuous-time mathematical law (``drift``/``diffusion``)
    from the discrete numerical scheme (``evolve``/``apply``).
    """

    def size(self) -> int:
        """Dimensionality of the state vector."""
        ...

    def factors(self) -> int:
        """Number of independent Brownian motions."""
        ...

    def initial_values(self) -> NDArray[np.float64]:
        """Starting state x(0)."""
        ...

    def drift(self, t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Continuous-time drift μ(t, x). Shape: ``(size,)``."""
        ...

    def diffusion(self, t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Continuous-time diffusion σ(t, x). Shape: ``(size, factors)``."""
        ...

    def evolve(
        self,
        t0: float,
        x0: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Advance state one step given noise increment *dw*."""
        ...


@runtime_checkable
class Discretization(Protocol):
    """Pluggable discretisation strategy for SDE integration."""

    def evolve(
        self,
        process: StochasticProcess,
        t0: float,
        x0: NDArray[np.float64],
        dt: float,
        dw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Advance *process* one step using this scheme."""
        ...


@runtime_checkable
class PortfolioGenerator(Protocol):
    """Protocol for portfolio generation strategies."""

    @property
    def code(self) -> str:
        """Short identifier used for registry look-up."""
        ...

    def generate(self, market_weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return portfolio weights given current market weights."""
        ...
