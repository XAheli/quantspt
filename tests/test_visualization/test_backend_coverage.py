"""Coverage tests for visualization/_backend.py — import error paths."""

from __future__ import annotations

import sys

import pytest

from quantspt.visualization._backend import (
    _get_matplotlib,
    _get_plotly,
    _validate_backend,
)


class TestGetPlotly:
    def test_success(self) -> None:
        go = _get_plotly()
        assert hasattr(go, "Figure")

    def test_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "plotly", None)
        monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
        with pytest.raises(ImportError, match="pip install quantspt"):
            _get_plotly()


class TestGetMatplotlib:
    def test_success(self) -> None:
        plt, Figure = _get_matplotlib()
        assert hasattr(plt, "subplots")
        assert Figure is not None

    def test_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
        with pytest.raises(ImportError, match="pip install quantspt"):
            _get_matplotlib()


class TestValidateBackend:
    def test_plotly_valid(self) -> None:
        assert _validate_backend("plotly") == "plotly"

    def test_matplotlib_valid(self) -> None:
        assert _validate_backend("matplotlib") == "matplotlib"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="'plotly' or 'matplotlib'"):
            _validate_backend("bokeh")
