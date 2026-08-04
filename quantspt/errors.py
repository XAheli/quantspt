"""Rich domain exception hierarchy for quantspt.

Every exception carries enough context for the caller to understand *what*
invariant was violated, *where* in the mathematical framework it belongs, and
(when possible) *how* to fix the issue.
"""

from __future__ import annotations


class SPTError(Exception):
    """Base exception for all quantspt errors."""


# -- mathematical invariants --------------------------------------------------


class SPTInvariantError(SPTError):
    """A mathematical invariant was violated.

    Examples: weights do not sum to 1, covariance matrix is not PSD,
    diversity parameter out of range.
    """


class DiversityConditionError(SPTError):
    """The required diversity condition is not satisfied.

    Raised when strict or weak diversity checks fail and the downstream
    computation requires diversity (e.g., arbitrage horizon estimation).
    """


class NumericalInstabilityError(SPTError):
    """Computation produced NaN, Inf, or loss of significance.

    Typically triggered by near-singular covariance matrices, extreme market
    weights, or aggressive discretisation step sizes.
    """


# -- simulation ---------------------------------------------------------------


class SimulationDivergenceError(SPTError):
    """SDE simulation diverged (NaN, Inf, or state left valid domain)."""


# -- estimation & calibration -------------------------------------------------


class CalibrationError(SPTError):
    """Model calibration failed to converge."""


class EstimationError(SPTError):
    """Statistical estimation produced an unreliable result."""


# -- optimisation -------------------------------------------------------------


class OptimizationError(SPTError):
    """Solver returned a non-optimal status."""


class InfeasibleError(OptimizationError):
    """The optimisation problem has no feasible solution."""


# -- data layer ---------------------------------------------------------------


class DataProviderError(SPTError):
    """Data provider failed to fetch or transform data."""


class SchemaValidationError(DataProviderError):
    """Data does not conform to the expected schema."""
