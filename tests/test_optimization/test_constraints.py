"""Tests for optimization/constraints -- constraint builders.

Validates ConstraintSet composition, verification, and CVXPY conversion.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.optimization.constraints import ConstraintSet


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestConstraintSet:
    """Tests for the ConstraintSet builder."""

    def test_position_limits_verify(self) -> None:
        cs = ConstraintSet()
        cs.add_position_limits(0.0, 0.5)
        assert cs.verify(np.array([0.3, 0.3, 0.4]))
        assert not cs.verify(np.array([0.6, 0.2, 0.2]))

    def test_negative_weight_rejected(self) -> None:
        cs = ConstraintSet()
        cs.add_position_limits(0.0, 1.0)
        assert not cs.verify(np.array([-0.1, 0.6, 0.5]))

    def test_turnover_verify(self) -> None:
        prev = np.array([0.5, 0.3, 0.2])
        cs = ConstraintSet()
        cs.add_turnover(0.1, prev)

        close = np.array([0.45, 0.35, 0.20])
        far = np.array([0.1, 0.1, 0.8])

        assert cs.verify(close)
        assert not cs.verify(far)

    def test_sector_constraints_verify(self) -> None:
        cs = ConstraintSet()
        cs.add_sector_constraints(
            sector_map={"tech": [0, 1], "fin": [2, 3]},
            sector_bounds={"tech": (0.2, 0.6), "fin": (0.2, 0.6)},
        )
        good = np.array([0.2, 0.2, 0.3, 0.3])
        assert cs.verify(good)

        bad_tech = np.array([0.4, 0.4, 0.1, 0.1])
        assert not cs.verify(bad_tech)

    def test_custom_constraint(self) -> None:
        cs = ConstraintSet()
        cs.add_custom(lambda w: float(w[0]) < 0.3)
        assert cs.verify(np.array([0.2, 0.4, 0.4]))
        assert not cs.verify(np.array([0.5, 0.3, 0.2]))

    def test_chaining(self) -> None:
        """Builder methods return self for chaining."""
        prev = np.array([0.5, 0.5])
        cs = ConstraintSet().add_position_limits(0.1, 0.9).add_turnover(0.3, prev)
        assert cs.min_weight == 0.1
        assert cs.max_weight == 0.9
        assert cs.max_turnover == 0.3

    def test_to_cvxpy_produces_constraints(self) -> None:
        """to_cvxpy should produce a non-empty list of constraints."""
        import cvxpy as cp

        cs = ConstraintSet()
        cs.add_position_limits(0.0, 0.5)
        pi = cp.Variable(3)
        constraints = cs.to_cvxpy(pi)
        assert len(constraints) >= 2

    def test_sector_cvxpy_constraints(self) -> None:
        import cvxpy as cp

        cs = ConstraintSet()
        cs.add_sector_constraints(
            sector_map={"tech": [0, 1]},
            sector_bounds={"tech": (0.3, 0.6)},
        )
        pi = cp.Variable(3)
        constraints = cs.to_cvxpy(pi)
        assert len(constraints) == 2

    def test_empty_constraint_set_verifies_all(self) -> None:
        """Empty ConstraintSet should accept any weights."""
        cs = ConstraintSet()
        assert cs.verify(np.array([0.5, 0.3, 0.2]))
        assert cs.verify(np.array([-0.1, 0.6, 0.5]))


class TestConstraintHelpers:
    """Tests for standalone constraint helper functions."""

    def test_position_limit_dict(self) -> None:
        from quantspt.optimization.constraints import position_limit_constraints

        result = position_limit_constraints(0.05, 0.3)
        assert result["min_weight"] == 0.05
        assert result["max_weight"] == 0.3

    def test_turnover_dict(self) -> None:
        from quantspt.optimization.constraints import turnover_constraint

        prev = np.array([0.5, 0.5])
        result = turnover_constraint(0.2, prev)
        assert result["max_turnover"] == 0.2
        assert_allclose(result["prev_weights"], prev)

    def test_sector_constraints_standalone(self) -> None:
        """Test the standalone sector_constraints function."""
        import cvxpy as cp

        from quantspt.optimization.constraints import sector_constraints

        pi = cp.Variable(4)
        constraints = sector_constraints(
            pi,
            sector_map={"tech": [0, 1], "fin": [2, 3]},
            sector_bounds={"tech": (0.2, 0.5), "fin": (0.3, 0.7)},
        )
        assert len(constraints) == 4

    def test_sector_constraints_missing_sector(self) -> None:
        """Sectors not in bounds are skipped."""
        import cvxpy as cp

        from quantspt.optimization.constraints import sector_constraints

        pi = cp.Variable(4)
        constraints = sector_constraints(
            pi,
            sector_map={"tech": [0, 1], "energy": [2, 3]},
            sector_bounds={"tech": (0.2, 0.5)},
        )
        assert len(constraints) == 2

    def test_to_cvxpy_with_turnover(self) -> None:
        """Test ConstraintSet.to_cvxpy with turnover constraint."""
        import cvxpy as cp

        cs = ConstraintSet()
        prev = np.array([0.4, 0.3, 0.3])
        cs.add_turnover(0.15, prev)
        pi = cp.Variable(3)
        constraints = cs.to_cvxpy(pi)
        assert len(constraints) >= 1
