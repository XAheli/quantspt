"""Coverage tests for visualization/export.py — all format branches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestToLatex:
    def test_exports_png(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        from quantspt.visualization.export import to_latex

        out = to_latex(fig, tmp_path / "fig.png")
        assert out.exists()
        assert out.suffix == ".png"

    def test_exports_svg(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        from quantspt.visualization.export import to_latex

        out = to_latex(fig, tmp_path / "fig.svg")
        assert out.exists()
        assert out.suffix == ".svg"

    def test_exports_eps(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        from quantspt.visualization.export import to_latex

        out = to_latex(fig, tmp_path / "fig.eps")
        assert out.exists()

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        from quantspt.errors import SPTInvariantError
        from quantspt.visualization.export import to_latex

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        with pytest.raises(SPTInvariantError):
            to_latex(fig, tmp_path / "fig.bmp")
        plt.close(fig)


class TestToPdf:
    def test_pdf_matplotlib(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        from quantspt.visualization.export import to_pdf

        out = to_pdf(fig, tmp_path / "fig.pdf")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_pdf_plotly(self, tmp_path: Path) -> None:
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 9])])

        from quantspt.visualization.export import to_pdf

        out = to_pdf(fig, tmp_path / "fig.pdf")
        assert out.exists()
        assert out.stat().st_size > 0


class TestToHtml:
    def test_html_plotly(self, tmp_path: Path) -> None:
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 9])])

        from quantspt.visualization.export import to_html

        out = to_html(fig, tmp_path / "fig.html")
        assert out.exists()
        content = out.read_text()
        assert "plotly" in content.lower() or "scatter" in content.lower()

    def test_html_matplotlib_full(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        from quantspt.visualization.export import to_html

        out = to_html(fig, tmp_path / "fig.html", full_html=True)
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "<svg" in content

    def test_html_matplotlib_not_full(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        from quantspt.visualization.export import to_html

        out = to_html(fig, tmp_path / "fig.html", full_html=False)
        assert out.exists()
        content = out.read_text()
        assert "<svg" in content
        assert "<!DOCTYPE html>" not in content


class TestGenerateReport:
    def _backtest_result(self) -> dict:
        rng = np.random.default_rng(42)
        return {
            "returns": rng.standard_normal(100) * 0.01,
            "metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08,
                "n_trades": 50,
            },
            "weights": {
                "AAPL": 0.30,
                "GOOGL": 0.25,
                "MSFT": 0.20,
                "AMZN": 0.15,
                "META": 0.10,
            },
        }

    def test_html_report(self, tmp_path: Path) -> None:
        from quantspt.visualization.export import generate_report

        result = self._backtest_result()
        out = generate_report(result, tmp_path / "report.html", format="html")
        assert out.exists()
        content = out.read_text()
        assert "Backtest Report" in content
        assert "Performance Metrics" in content
        assert "Cumulative Returns" in content
        assert "AAPL" in content

    def test_html_report_minimal(self, tmp_path: Path) -> None:
        from quantspt.visualization.export import generate_report

        result = {"returns": np.random.default_rng(42).standard_normal(50) * 0.01}
        out = generate_report(
            result,
            tmp_path / "report.html",
            format="html",
            include_metrics=False,
            include_weights=False,
        )
        assert out.exists()

    def test_latex_report(self, tmp_path: Path) -> None:
        from quantspt.visualization.export import generate_report

        result = self._backtest_result()
        out = generate_report(result, tmp_path / "report.tex", format="latex")
        assert out.exists()
        content = out.read_text()
        assert r"\documentclass" in content
        assert r"\begin{tabular}" in content

    def test_latex_report_no_metrics(self, tmp_path: Path) -> None:
        from quantspt.visualization.export import generate_report

        result = {"returns": np.array([0.01, -0.02, 0.03])}
        out = generate_report(
            result,
            tmp_path / "report.tex",
            format="latex",
            include_metrics=False,
        )
        assert out.exists()
        content = out.read_text()
        assert r"\begin{tabular}" not in content

    def test_pdf_report(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")

        from quantspt.visualization.export import generate_report

        result = self._backtest_result()
        out = generate_report(result, tmp_path / "report.pdf", format="pdf")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_pdf_report_no_charts(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")

        from quantspt.visualization.export import generate_report

        result = self._backtest_result()
        out = generate_report(
            result,
            tmp_path / "report.pdf",
            format="pdf",
            include_charts=False,
        )
        assert out.exists()

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        from quantspt.errors import SPTInvariantError
        from quantspt.visualization.export import generate_report

        with pytest.raises(SPTInvariantError):
            generate_report({}, tmp_path / "report.docx", format="docx")

    def test_normalize_result_object(self, tmp_path: Path) -> None:
        """Test _normalize_result with an object instead of dict."""
        from quantspt.visualization.export import generate_report

        class FakeResult:
            def __init__(self) -> None:
                self.returns = np.random.default_rng(42).standard_normal(50) * 0.01
                self.metrics = {"sharpe": 1.0}

        out = generate_report(FakeResult(), tmp_path / "report.html", format="html")
        assert out.exists()

    def test_returns_chart_with_series_values(self, tmp_path: Path) -> None:
        """Returns with a .values attribute (like pandas Series)."""
        from quantspt.visualization.export import generate_report

        class FakeSeries:
            def __init__(self, data: np.ndarray) -> None:
                self.values = data

            def __len__(self) -> int:
                return len(self.values)

        result = {
            "returns": FakeSeries(
                np.random.default_rng(42).standard_normal(50).reshape(-1, 1) * 0.01
            )
        }
        out = generate_report(result, tmp_path / "report.html", format="html")
        assert out.exists()

    def test_weights_with_dataframe_columns(self, tmp_path: Path) -> None:
        """Weights with .columns and .iloc (like pandas DataFrame)."""
        import pandas as pd

        from quantspt.visualization.export import generate_report

        weights_df = pd.DataFrame(
            {"AAPL": [0.3, 0.35], "GOOGL": [0.7, 0.65]}, index=[0, 1]
        )
        result = {
            "returns": np.array([0.01, 0.02]),
            "weights": weights_df,
        }
        out = generate_report(result, tmp_path / "report.html", format="html")
        assert out.exists()
        content = out.read_text()
        assert "AAPL" in content

    def test_weights_returns_empty_for_array(self, tmp_path: Path) -> None:
        """Weights as an array (not dict or df) yields empty string."""
        from quantspt.visualization.export import generate_report

        result = {
            "returns": np.array([0.01, 0.02]),
            "weights": np.array([[0.5, 0.5], [0.6, 0.4]]),
        }
        out = generate_report(result, tmp_path / "report.html", format="html")
        assert out.exists()

    def test_pdf_with_series_values(self, tmp_path: Path) -> None:
        """PDF report with returns having .values (pandas-like)."""
        import matplotlib

        matplotlib.use("Agg")

        from quantspt.visualization.export import generate_report

        class FakeSeries:
            def __init__(self, data: np.ndarray) -> None:
                self.values = data

            def __len__(self) -> int:
                return len(self.values)

        result = {
            "returns": FakeSeries(
                np.random.default_rng(42).standard_normal(50).reshape(-1, 1) * 0.01
            ),
            "metrics": {"sharpe": 1.0},
        }
        out = generate_report(result, tmp_path / "report.pdf", format="pdf")
        assert out.exists()


class TestExportImportErrors:
    """Test import error paths for matplotlib/plotly."""

    def test_require_matplotlib_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from quantspt.visualization.export import _require_matplotlib

        monkeypatch.setitem(sys.modules, "matplotlib", None)
        with pytest.raises(ImportError, match="pip install quantspt"):
            _require_matplotlib()

    def test_require_plotly_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from quantspt.visualization.export import _require_plotly

        monkeypatch.setitem(sys.modules, "plotly", None)
        with pytest.raises(ImportError, match="pip install quantspt"):
            _require_plotly()

    def test_is_plotly_figure_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        from quantspt.visualization.export import _is_plotly_figure

        monkeypatch.setitem(sys.modules, "plotly", None)
        monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
        assert _is_plotly_figure("not a figure") is False
