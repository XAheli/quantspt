"""Contract-based design helpers for mathematical invariant enforcement.

These lightweight guards enforce mathematical invariants at function boundaries,
producing rich error messages that reference the violated condition.
"""

from __future__ import annotations

from .errors import SPTInvariantError


def require(condition: bool, message: str) -> None:
    """Check a precondition; raise on violation.

    Use at function entry to validate inputs against mathematical requirements
    (e.g., weights sum to 1, covariance matrix is PSD).

    Parameters
    ----------
    condition
        Boolean expression that must be ``True``.
    message
        Human-readable description of the requirement.

    Raises
    ------
    SPTInvariantError
        When *condition* is ``False``.
    """
    if not condition:
        raise SPTInvariantError(f"Precondition failed: {message}")


def ensure(condition: bool, message: str) -> None:
    """Check a postcondition; raise on violation.

    Use after computation to verify that results satisfy expected mathematical
    properties (e.g., returned weights are non-negative).

    Parameters
    ----------
    condition
        Boolean expression that must be ``True``.
    message
        Human-readable description of the guarantee.

    Raises
    ------
    SPTInvariantError
        When *condition* is ``False``.
    """
    if not condition:
        raise SPTInvariantError(f"Postcondition failed: {message}")
