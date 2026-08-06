"""Tests for data/universe.py — Universe construction and membership."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantspt.data.schemas import MarketPanel
from quantspt.data.universe import Universe, reconstruct
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_dates():
    return pd.date_range("2020-01-01", periods=10, freq="B")


@pytest.fixture()
def sample_prices(sample_dates):
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        rng.uniform(50, 150, (10, 3)),
        index=sample_dates,
        columns=["A", "B", "C"],
    )
    return prices


@pytest.fixture()
def panel_with_gaps(sample_dates):
    """Panel where asset B has NaN gaps."""
    prices = pd.DataFrame(
        {
            "A": np.arange(100.0, 110.0),
            "B": [50.0, np.nan, np.nan, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0],
            "C": [
                200.0,
                201.0,
                202.0,
                203.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        },
        index=sample_dates,
    )
    return MarketPanel(prices=prices, tickers=["A", "B", "C"])


@pytest.fixture()
def panel_with_caps(sample_dates):
    """Panel with market cap data."""
    prices = pd.DataFrame(
        {"X": np.arange(10.0, 20.0), "Y": np.arange(100.0, 110.0)},
        index=sample_dates,
    )
    caps = pd.DataFrame(
        {"X": [1e6] * 5 + [1e9] * 5, "Y": [1e9] * 10},
        index=sample_dates,
    )
    return MarketPanel(prices=prices, tickers=["X", "Y"], market_caps=caps)


# ---------------------------------------------------------------------------
# Universe class tests
# ---------------------------------------------------------------------------


class TestUniverse:
    def test_construction(self, sample_dates) -> None:
        membership = pd.DataFrame(
            {"A": [True] * 10, "B": [False] * 5 + [True] * 5},
            index=sample_dates,
        )
        u = Universe(membership=membership, tickers=["A", "B"])
        assert u.n_dates == 10
        assert u.n_assets == 2

    def test_invalid_non_bool_raises(self, sample_dates) -> None:
        membership = pd.DataFrame({"A": [1.0] * 10}, index=sample_dates)
        with pytest.raises(SPTInvariantError, match="boolean"):
            Universe(membership=membership, tickers=["A"])

    def test_ticker_mismatch_raises(self, sample_dates) -> None:
        membership = pd.DataFrame({"A": [True] * 10}, index=sample_dates)
        with pytest.raises(SPTInvariantError, match="Tickers"):
            Universe(membership=membership, tickers=["B"])

    def test_members_at(self, sample_dates) -> None:
        membership = pd.DataFrame(
            {"A": [True, True, False], "B": [False, True, True]},
            index=sample_dates[:3],
        )
        u = Universe(membership=membership, tickers=["A", "B"])
        assert u.members_at(sample_dates[0]) == ["A"]
        assert set(u.members_at(sample_dates[1])) == {"A", "B"}
        assert u.members_at(sample_dates[2]) == ["B"]

    def test_member_count(self, sample_dates) -> None:
        membership = pd.DataFrame(
            {"A": [True, True, False], "B": [False, True, True]},
            index=sample_dates[:3],
        )
        u = Universe(membership=membership, tickers=["A", "B"])
        counts = u.member_count()
        assert list(counts.values) == [1, 2, 1]

    def test_entry_dates(self, sample_dates) -> None:
        membership = pd.DataFrame(
            {"A": [True] * 5, "B": [False, False, True, True, True]},
            index=sample_dates[:5],
        )
        u = Universe(membership=membership, tickers=["A", "B"])
        entries = u.entry_dates()
        assert entries["A"] == sample_dates[0]
        assert entries["B"] == sample_dates[2]

    def test_exit_dates(self, sample_dates) -> None:
        membership = pd.DataFrame(
            {"A": [True, True, True, False, False], "B": [True] * 5},
            index=sample_dates[:5],
        )
        u = Universe(membership=membership, tickers=["A", "B"])
        exits = u.exit_dates()
        assert exits["A"] == sample_dates[2]
        assert exits["B"] == sample_dates[4]

    def test_apply_to_panel(self, sample_dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0], "B": [50.0, 51.0, 52.0]},
            index=sample_dates[:3],
        )
        panel = MarketPanel(prices=prices, tickers=["A", "B"])
        membership = pd.DataFrame(
            {"A": [True, True, False], "B": [True, False, True]},
            index=sample_dates[:3],
        )
        u = Universe(membership=membership, tickers=["A", "B"])
        result = u.apply_to_panel(panel)
        assert result.loc[sample_dates[0], "A"] == 100.0
        assert np.isnan(result.loc[sample_dates[2], "A"])
        assert np.isnan(result.loc[sample_dates[1], "B"])
        assert result.loc[sample_dates[2], "B"] == 52.0

    def test_turnover(self, sample_dates) -> None:
        membership = pd.DataFrame(
            {
                "A": [False, True, True, False],
                "B": [True, True, False, False],
            },
            index=sample_dates[:4],
        )
        u = Universe(membership=membership, tickers=["A", "B"])
        to = u.turnover()
        assert to["entries"].iloc[1] == 1  # A enters
        assert to["exits"].iloc[2] == 1  # B exits
        assert to["exits"].iloc[3] == 1  # A exits

    def test_never_member_entry_exit(self, sample_dates) -> None:
        membership = pd.DataFrame(
            {"A": [True] * 5, "B": [False] * 5},
            index=sample_dates[:5],
        )
        u = Universe(membership=membership, tickers=["A", "B"])
        assert u.entry_dates()["B"] is None
        assert u.exit_dates()["B"] is None


# ---------------------------------------------------------------------------
# reconstruct() tests
# ---------------------------------------------------------------------------


class TestReconstruct:
    def test_basic_reconstruction(self, panel_with_gaps) -> None:
        universe = reconstruct(panel_with_gaps, min_observations=1)
        assert isinstance(universe, Universe)
        assert universe.n_assets == 3
        assert universe.n_dates == 10

    def test_min_observations_filters(self, panel_with_gaps) -> None:
        universe = reconstruct(panel_with_gaps, min_observations=3)
        assert not universe.membership.iloc[0]["A"]
        assert not universe.membership.iloc[0]["B"]
        assert universe.membership.iloc[2]["A"]
        assert not universe.membership.iloc[2]["B"]  # B has gaps

    def test_min_observations_one(self, panel_with_gaps) -> None:
        universe = reconstruct(panel_with_gaps, min_observations=1)
        assert universe.membership.iloc[0]["A"]
        assert universe.membership.iloc[0]["B"]

    def test_nan_excludes_membership(self, panel_with_gaps) -> None:
        universe = reconstruct(panel_with_gaps, min_observations=1)
        assert not universe.membership.iloc[1]["B"]  # B is NaN at index 1
        assert not universe.membership.iloc[2]["B"]  # B is NaN at index 2

    def test_min_market_cap(self, panel_with_caps) -> None:
        universe = reconstruct(panel_with_caps, min_observations=1, min_market_cap=1e8)
        assert not universe.membership.iloc[0]["X"]
        assert universe.membership.iloc[5]["X"]
        assert universe.membership.iloc[0]["Y"]

    def test_min_price(self, sample_dates) -> None:
        prices = pd.DataFrame(
            {"A": [5.0, 15.0, 25.0], "B": [50.0, 50.0, 50.0]},
            index=sample_dates[:3],
        )
        panel = MarketPanel(prices=prices, tickers=["A", "B"])
        universe = reconstruct(panel, min_observations=1, min_price=10.0)
        assert not universe.membership.iloc[0]["A"]
        assert universe.membership.iloc[1]["A"]
        assert universe.membership.iloc[0]["B"]

    def test_lookback_window(self, sample_dates) -> None:
        prices = pd.DataFrame(
            {"A": list(range(1, 11))},
            index=sample_dates,
            dtype=float,
        )
        panel = MarketPanel(prices=prices, tickers=["A"])
        universe = reconstruct(panel, min_observations=3, lookback_window=3)
        assert not universe.membership.iloc[0]["A"]
        assert not universe.membership.iloc[1]["A"]
        assert universe.membership.iloc[2]["A"]

    def test_metadata_stored(self, panel_with_gaps) -> None:
        universe = reconstruct(
            panel_with_gaps,
            min_observations=5,
            min_price=10.0,
            lookback_window=20,
        )
        assert universe.metadata["min_observations"] == 5
        assert universe.metadata["min_price"] == 10.0
        assert universe.metadata["lookback_window"] == 20

    def test_invalid_min_observations_raises(self, panel_with_gaps) -> None:
        with pytest.raises(SPTInvariantError):
            reconstruct(panel_with_gaps, min_observations=0)

    def test_time_varying_membership(self, sample_dates) -> None:
        """Assets should enter when they have enough data."""
        prices = pd.DataFrame(
            {
                "A": [100.0] * 10,
                "B": [np.nan] * 4 + [50.0] * 6,
            },
            index=sample_dates,
        )
        panel = MarketPanel(prices=prices, tickers=["A", "B"])
        universe = reconstruct(panel, min_observations=2)
        assert universe.membership.iloc[3]["A"]
        assert not universe.membership.iloc[4]["B"]
        assert universe.membership.iloc[5]["B"]

    def test_mostly_nan_asset_enters_late(self, sample_dates) -> None:
        """Asset with mostly NaN data only enters once it has enough observations."""
        prices = pd.DataFrame(
            {"A": [100.0] * 10, "B": [np.nan] * 8 + [50.0, 51.0]},
            index=sample_dates,
        )
        panel = MarketPanel(prices=prices, tickers=["A", "B"])
        universe = reconstruct(panel, min_observations=2)
        assert not universe.membership.iloc[8]["B"]
        assert universe.membership.iloc[9]["B"]
