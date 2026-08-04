"""Tests for _result.py — SPTResult container and timed_result context manager."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from quantspt._result import SPTResult, timed_result

# =========================================================================
# SPTResult construction and defaults
# =========================================================================


class TestSPTResultConstruction:
    def test_default_fields(self) -> None:
        result: SPTResult[str] = SPTResult(data="hello")
        assert result.data == "hello"
        assert result.metadata == {}
        assert result.warnings == []
        assert result.computation_time_ms == 0.0

    def test_with_all_fields(self) -> None:
        result = SPTResult(
            data=np.array([1.0, 2.0]),
            metadata={"method": "exact"},
            warnings=["low sample"],
            computation_time_ms=42.5,
        )
        assert result.metadata["method"] == "exact"
        assert len(result.warnings) == 1
        assert result.computation_time_ms == 42.5


# =========================================================================
# .to_dataframe()
# =========================================================================


class TestToDataframe:
    def test_from_ndarray(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = SPTResult(data=arr)
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert df.iloc[0, 0] == 1.0
        assert df.iloc[1, 1] == 4.0

    def test_from_dataframe(self) -> None:
        """If data is already a DataFrame, return it directly."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = SPTResult(data=df)
        out = result.to_dataframe()
        assert out is df

    def test_from_1d_array(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = SPTResult(data=arr)
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 1)

    def test_raises_on_invalid_type(self) -> None:
        result = SPTResult(data="not-an-array")
        with pytest.raises(TypeError, match="Cannot convert"):
            result.to_dataframe()


# =========================================================================
# .metadata access
# =========================================================================


class TestMetadata:
    def test_metadata_access(self) -> None:
        result = SPTResult(
            data=None,
            metadata={"solver": "ECOS", "iterations": 15},
        )
        assert result.metadata["solver"] == "ECOS"
        assert result.metadata["iterations"] == 15

    def test_empty_metadata(self) -> None:
        result = SPTResult(data=None)
        assert result.metadata == {}


# =========================================================================
# .warnings
# =========================================================================


class TestWarnings:
    def test_warnings_list(self) -> None:
        result = SPTResult(
            data=None,
            warnings=["Near-singular covariance", "Low sample size"],
        )
        assert len(result.warnings) == 2
        assert "Near-singular" in result.warnings[0]

    def test_empty_warnings(self) -> None:
        result = SPTResult(data=None)
        assert result.warnings == []


# =========================================================================
# .validate()
# =========================================================================


class TestValidate:
    def test_valid_array(self) -> None:
        result = SPTResult(data=np.array([1.0, 2.0, 3.0]))
        assert result.validate() is True

    def test_nan_data_invalid(self) -> None:
        result = SPTResult(data=np.array([1.0, np.nan, 3.0]))
        assert result.validate() is False

    def test_inf_data_invalid(self) -> None:
        result = SPTResult(data=np.array([1.0, np.inf, 3.0]))
        assert result.validate() is False

    def test_neginf_data_invalid(self) -> None:
        result = SPTResult(data=np.array([1.0, -np.inf, 3.0]))
        assert result.validate() is False

    def test_non_array_data_valid(self) -> None:
        result = SPTResult(data="just a string")
        assert result.validate() is True

    def test_none_data_valid(self) -> None:
        result = SPTResult(data=None)
        assert result.validate() is True


# =========================================================================
# __repr__
# =========================================================================


class TestRepr:
    def test_repr_format(self) -> None:
        result = SPTResult(
            data=np.array([1.0]),
            warnings=["w1", "w2"],
            computation_time_ms=12.3,
        )
        r = repr(result)
        assert "SPTResult" in r
        assert "ndarray" in r
        assert "warnings=2" in r
        assert "12.3ms" in r

    def test_repr_no_warnings(self) -> None:
        result = SPTResult(data=42, computation_time_ms=0.5)
        r = repr(result)
        assert "warnings=0" in r
        assert "int" in r


# =========================================================================
# .summary()
# =========================================================================


class TestSummary:
    def test_summary_basic(self) -> None:
        result = SPTResult(data=np.array([1.0]), computation_time_ms=5.0)
        s = result.summary()
        assert "SPTResult" in s
        assert "5.0 ms" in s

    def test_summary_with_warnings(self) -> None:
        result = SPTResult(data=None, warnings=["w1"], computation_time_ms=1.0)
        s = result.summary()
        assert "warnings" in s

    def test_summary_verbose(self) -> None:
        result = SPTResult(
            data=None,
            metadata={"solver": "ECOS"},
            computation_time_ms=1.0,
        )
        s = result.summary(verbose=True)
        assert "metadata" in s
        assert "ECOS" in s

    def test_summary_verbose_no_metadata(self) -> None:
        result = SPTResult(data=None, computation_time_ms=1.0)
        s = result.summary(verbose=True)
        assert "SPTResult" in s


# =========================================================================
# .to_json()
# =========================================================================


class TestToJson:
    def test_json_output(self) -> None:
        result = SPTResult(
            data=np.array([1.0]),
            metadata={"method": "exact"},
            warnings=["w1"],
            computation_time_ms=10.0,
        )
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["metadata"]["method"] == "exact"
        assert parsed["warnings"] == ["w1"]
        assert parsed["computation_time_ms"] == 10.0

    def test_json_excludes_data(self) -> None:
        result = SPTResult(data=np.array([1.0, 2.0]))
        parsed = json.loads(result.to_json())
        assert "data" not in parsed


# =========================================================================
# timed_result context manager
# =========================================================================


class TestTimedResult:
    def test_measures_elapsed(self) -> None:
        with timed_result() as tr:
            time.sleep(0.01)
        assert tr.elapsed_ms > 5.0

    def test_initial_zero(self) -> None:
        tr = timed_result()
        assert tr.elapsed_ms == 0.0

    def test_context_manager_returns_self(self) -> None:
        cm = timed_result()
        with cm as tr:
            assert tr is cm
