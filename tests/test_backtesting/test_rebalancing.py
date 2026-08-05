"""Tests for backtesting rebalancing triggers.

Validates calendar, threshold, and drift-based rebalancing logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.backtesting.rebalancing import (
    CalendarRebalancer,
    DriftRebalancer,
    Frequency,
    Rebalancer,
    ThresholdRebalancer,
)
from quantspt.errors import SPTInvariantError

# =========================================================================
# Protocol conformance
# =========================================================================


class TestRebalancerProtocol:
    """All concrete rebalancers must satisfy the Rebalancer protocol."""

    def test_calendar_is_rebalancer(self) -> None:
        r = CalendarRebalancer(Frequency.MONTHLY)
        assert isinstance(r, Rebalancer)

    def test_threshold_is_rebalancer(self) -> None:
        r = ThresholdRebalancer(0.05)
        assert isinstance(r, Rebalancer)

    def test_drift_is_rebalancer(self) -> None:
        r = DriftRebalancer(0.1)
        assert isinstance(r, Rebalancer)


# =========================================================================
# CalendarRebalancer
# =========================================================================


class TestCalendarRebalancer:
    """Tests for fixed-schedule rebalancing."""

    def test_daily_triggers_every_step(self) -> None:
        r = CalendarRebalancer(Frequency.DAILY)
        w = np.array([0.5, 0.5])
        for step in range(10):
            assert r.should_rebalance(step, w, w) is True

    def test_weekly_triggers_every_5_steps(self) -> None:
        r = CalendarRebalancer(Frequency.WEEKLY)
        w = np.array([0.5, 0.5])
        expected = {0, 5, 10, 15, 20}
        for step in range(25):
            result = r.should_rebalance(step, w, w)
            if step in expected:
                assert result is True, f"Should rebalance at step {step}"
            else:
                assert result is False, f"Should NOT rebalance at step {step}"

    def test_monthly_triggers_every_21_steps(self) -> None:
        r = CalendarRebalancer(Frequency.MONTHLY)
        w = np.array([0.3, 0.3, 0.4])
        assert r.should_rebalance(0, w, w) is True
        assert r.should_rebalance(1, w, w) is False
        assert r.should_rebalance(20, w, w) is False
        assert r.should_rebalance(21, w, w) is True
        assert r.should_rebalance(42, w, w) is True

    def test_quarterly_triggers_every_63_steps(self) -> None:
        r = CalendarRebalancer(Frequency.QUARTERLY)
        w = np.array([0.5, 0.5])
        assert r.should_rebalance(0, w, w) is True
        assert r.should_rebalance(62, w, w) is False
        assert r.should_rebalance(63, w, w) is True
        assert r.should_rebalance(126, w, w) is True

    def test_ignores_weight_values(self) -> None:
        """Calendar rebalancer doesn't depend on weight values."""
        r = CalendarRebalancer(Frequency.MONTHLY)
        w1 = np.array([0.1, 0.9])
        w2 = np.array([0.5, 0.5])
        assert r.should_rebalance(21, w1, w2) is True
        assert r.should_rebalance(10, w1, w2) is False

    def test_frequency_values(self) -> None:
        """Verify frequency enum values match trading calendar."""
        assert Frequency.DAILY.value == 1
        assert Frequency.WEEKLY.value == 5
        assert Frequency.MONTHLY.value == 21
        assert Frequency.QUARTERLY.value == 63


# =========================================================================
# ThresholdRebalancer
# =========================================================================


class TestThresholdRebalancer:
    """Tests for max-drift threshold rebalancing."""

    def test_always_rebalances_at_step_0(self) -> None:
        r = ThresholdRebalancer(0.05)
        current = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])
        assert r.should_rebalance(0, current, target) is True

    def test_no_rebalance_below_threshold(self) -> None:
        r = ThresholdRebalancer(0.05)
        current = np.array([0.52, 0.48])
        target = np.array([0.50, 0.50])
        assert r.should_rebalance(1, current, target) is False

    def test_rebalance_above_threshold(self) -> None:
        r = ThresholdRebalancer(0.05)
        current = np.array([0.56, 0.44])
        target = np.array([0.50, 0.50])
        assert r.should_rebalance(1, current, target) is True

    def test_just_below_threshold_no_trigger(self) -> None:
        """Drift clearly below threshold should NOT trigger."""
        r = ThresholdRebalancer(0.10)
        current = np.array([0.55, 0.45])
        target = np.array([0.50, 0.50])
        assert r.should_rebalance(1, current, target) is False

    def test_multi_asset_max_drift(self) -> None:
        """Uses max absolute drift across all assets."""
        r = ThresholdRebalancer(0.10)
        current = np.array([0.3, 0.3, 0.4])
        target = np.array([0.33, 0.33, 0.34])
        max_drift = np.max(np.abs(current - target))
        assert max_drift < 0.10
        assert r.should_rebalance(5, current, target) is False

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="Threshold must be in"):
            ThresholdRebalancer(0.0)
        with pytest.raises(SPTInvariantError, match="Threshold must be in"):
            ThresholdRebalancer(1.0)
        with pytest.raises(SPTInvariantError, match="Threshold must be in"):
            ThresholdRebalancer(-0.1)


# =========================================================================
# DriftRebalancer
# =========================================================================


class TestDriftRebalancer:
    """Tests for L2-drift-based rebalancing."""

    def test_always_rebalances_at_step_0(self) -> None:
        r = DriftRebalancer(0.1)
        w = np.array([0.5, 0.5])
        assert r.should_rebalance(0, w, w) is True

    def test_no_rebalance_below_drift(self) -> None:
        r = DriftRebalancer(0.1)
        current = np.array([0.52, 0.48])
        target = np.array([0.50, 0.50])
        l2 = float(np.linalg.norm(current - target))
        assert l2 < 0.1
        assert r.should_rebalance(1, current, target) is False

    def test_rebalance_above_drift(self) -> None:
        r = DriftRebalancer(0.05)
        current = np.array([0.55, 0.45])
        target = np.array([0.50, 0.50])
        l2 = float(np.linalg.norm(current - target))
        assert l2 > 0.05
        assert r.should_rebalance(1, current, target) is True

    def test_l2_norm_computation(self) -> None:
        """Verify L2 norm is used (not L1 or Linf)."""
        r = DriftRebalancer(0.10)
        current = np.array([0.35, 0.35, 0.30])
        target = np.array([0.30, 0.30, 0.40])
        l2 = float(np.linalg.norm(current - target))
        l1 = float(np.sum(np.abs(current - target)))
        assert l2 < l1  # sanity: L2 < L1 for multi-dim
        if l2 > 0.10:
            assert r.should_rebalance(1, current, target) is True

    def test_invalid_drift_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="max_drift must be positive"):
            DriftRebalancer(0.0)
        with pytest.raises(SPTInvariantError, match="max_drift must be positive"):
            DriftRebalancer(-0.05)
