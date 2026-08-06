"""Tests for post_processing/export.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantspt.post_processing.discrete_allocation import AllocationResult
from quantspt.post_processing.export import to_csv, to_dataframe, to_json


@pytest.fixture()
def sample_allocation() -> AllocationResult:
    return AllocationResult(
        shares=np.array([10, 20, 5], dtype=np.intp),
        leftover_cash=42.50,
        actual_weights=np.array([0.4, 0.35, 0.25]),
    )


@pytest.fixture()
def sample_weights() -> np.ndarray:
    return np.array([0.4, 0.35, 0.25])


class TestToDataframe:
    def test_allocation_result(self, sample_allocation) -> None:
        df = to_dataframe(sample_allocation)
        assert isinstance(df, pd.DataFrame)
        assert "ticker" in df.columns
        assert "shares" in df.columns
        assert "actual_weight" in df.columns
        assert len(df) == 3

    def test_weight_vector(self, sample_weights) -> None:
        df = to_dataframe(sample_weights)
        assert isinstance(df, pd.DataFrame)
        assert "ticker" in df.columns
        assert "weight" in df.columns
        assert len(df) == 3

    def test_custom_tickers(self, sample_allocation) -> None:
        tickers = ["AAPL", "GOOG", "MSFT"]
        df = to_dataframe(sample_allocation, tickers=tickers)
        assert list(df["ticker"]) == tickers

    def test_default_tickers(self, sample_allocation) -> None:
        df = to_dataframe(sample_allocation)
        assert df["ticker"].iloc[0] == "Asset_0"


class TestToCSV:
    def test_round_trip_allocation(self, tmp_path, sample_allocation) -> None:
        path = tmp_path / "test.csv"
        result_path = to_csv(sample_allocation, path)
        assert result_path.exists()

        df = pd.read_csv(result_path)
        assert len(df) == 3
        assert "shares" in df.columns
        np.testing.assert_array_equal(df["shares"].values, [10, 20, 5])

    def test_round_trip_weights(self, tmp_path, sample_weights) -> None:
        path = tmp_path / "weights.csv"
        result_path = to_csv(sample_weights, path)
        assert result_path.exists()

        df = pd.read_csv(result_path)
        assert len(df) == 3
        np.testing.assert_allclose(df["weight"].values, sample_weights, atol=1e-10)

    def test_custom_tickers_csv(self, tmp_path, sample_weights) -> None:
        path = tmp_path / "test.csv"
        tickers = ["SPY", "QQQ", "IWM"]
        to_csv(sample_weights, path, tickers=tickers)
        df = pd.read_csv(path)
        assert list(df["ticker"]) == tickers


class TestToJSON:
    def test_round_trip_allocation(self, tmp_path, sample_allocation) -> None:
        path = tmp_path / "test.json"
        result_path = to_json(sample_allocation, path)
        assert result_path.exists()

        data = json.loads(result_path.read_text())
        assert "allocations" in data
        assert "leftover_cash" in data
        assert len(data["allocations"]) == 3
        assert data["leftover_cash"] == pytest.approx(42.50)
        assert data["allocations"][0]["shares"] == 10

    def test_round_trip_weights(self, tmp_path, sample_weights) -> None:
        path = tmp_path / "weights.json"
        result_path = to_json(sample_weights, path)
        assert result_path.exists()

        data = json.loads(result_path.read_text())
        assert "weights" in data
        assert len(data["weights"]) == 3
        assert data["weights"][0]["weight"] == pytest.approx(0.4)

    def test_custom_tickers_json(self, tmp_path, sample_allocation) -> None:
        path = tmp_path / "test.json"
        tickers = ["TSLA", "NVDA", "AMD"]
        to_json(sample_allocation, path, tickers=tickers)
        data = json.loads(Path(path).read_text())
        assert data["allocations"][0]["ticker"] == "TSLA"
