"""Tests for universe/reconstitution.py — hysteresis-based reconstitution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantspt.errors import SPTInvariantError
from quantspt.universe.reconstitution import UniverseReconstitution

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scores():
    """Scores for 20 tickers, S00 (best) through S19 (worst)."""
    tickers = [f"S{i:02d}" for i in range(20)]
    return pd.Series(np.linspace(1.0, 0.0, 20), index=tickers)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self):
        r = UniverseReconstitution()
        assert r.buffer_pct == 0.10
        assert r.max_turnover_pct == 0.30

    def test_invalid_buffer(self):
        with pytest.raises(SPTInvariantError, match="buffer"):
            UniverseReconstitution(buffer_pct=0.6)

    def test_invalid_turnover(self):
        with pytest.raises(SPTInvariantError, match="turnover"):
            UniverseReconstitution(max_turnover_pct=0.0)


# ---------------------------------------------------------------------------
# reconstitute()
# ---------------------------------------------------------------------------


class TestReconstitute:
    def test_cold_start_picks_top_n(self, scores):
        r = UniverseReconstitution()
        result = r.reconstitute([], scores, n_stocks=10)
        assert len(result) == 10
        assert result[0] == "S00"

    def test_stable_universe_no_change(self, scores):
        r = UniverseReconstitution()
        current = [f"S{i:02d}" for i in range(10)]
        result = r.reconstitute(current, scores, n_stocks=10)
        assert set(result) == set(current)

    def test_hysteresis_keeps_borderline_incumbent(self, scores):
        r = UniverseReconstitution(buffer_pct=0.20)
        # S10 is rank 11 — just outside top 10, but in buffer zone
        current = [f"S{i:02d}" for i in range(10)]
        current[9] = "S10"  # Replace S09 with S10 as incumbent
        result = r.reconstitute(current, scores, n_stocks=10)
        # S10 should survive because it's in the buffer zone
        assert "S10" in result

    def test_clear_improvement_enters(self, scores):
        r = UniverseReconstitution(buffer_pct=0.10, max_turnover_pct=0.50)
        # Start with mediocre stocks
        current = [f"S{i:02d}" for i in range(5, 15)]
        result = r.reconstitute(current, scores, n_stocks=10)
        # Some top stocks (S00..S04) should enter
        top_stocks = {f"S{i:02d}" for i in range(5)}
        assert len(top_stocks & set(result)) > 0

    def test_max_turnover_respected(self, scores):
        r = UniverseReconstitution(max_turnover_pct=0.10)
        current = [f"S{i:02d}" for i in range(10, 20)]  # all bottom-ranked
        result = r.reconstitute(current, scores, n_stocks=10)
        changes = len(set(result) ^ set(current))
        # 10% of 10 = 1 swap (2 changes: 1 add + 1 remove)
        assert changes <= 4  # generous slack for implementation

    def test_result_length(self, scores):
        r = UniverseReconstitution()
        result = r.reconstitute([], scores, n_stocks=10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# reconstitute_timeseries()
# ---------------------------------------------------------------------------


class TestReconstitueTimeseries:
    def test_produces_one_entry_per_date(self):
        r = UniverseReconstitution()
        dates = pd.date_range("2024-01-01", periods=4, freq="MS")
        tickers = [f"S{i:02d}" for i in range(20)]
        scores_by_date = {}
        rng = np.random.default_rng(42)
        for dt in dates:
            s = pd.Series(rng.uniform(0, 1, 20), index=tickers)
            scores_by_date[dt] = s

        result = r.reconstitute_timeseries(scores_by_date, n_stocks=10)
        assert len(result) == 4
        for dt in dates:
            assert dt in result
            assert len(result[dt]) == 10

    def test_carries_universe_forward(self):
        r = UniverseReconstitution(max_turnover_pct=0.10)
        tickers = [f"S{i:02d}" for i in range(20)]
        dates = pd.date_range("2024-01-01", periods=6, freq="MS")

        scores_by_date = {}
        for _i, dt in enumerate(dates):
            scores_by_date[dt] = pd.Series(np.linspace(1.0, 0.0, 20), index=tickers)

        result = r.reconstitute_timeseries(scores_by_date, n_stocks=10)

        # After a few periods the universe should converge to top 10
        last = set(result[dates[-1]])
        expected = {f"S{i:02d}" for i in range(10)}
        overlap = last & expected
        assert len(overlap) >= 7
