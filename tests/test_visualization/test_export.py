"""Tests for visualization/export.py — report export utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantspt.errors import SPTInvariantError
from quantspt.visualization.export import generate_report, to_html, to_latex, to_pdf

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mpl_fig():
    """Create a simple matplotlib figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Test")
    return fig


@pytest.fixture()
def backtest_result():
    """Sample backtest result dict."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 252)
    weights = pd.DataFrame(
        {"A": np.linspace(0.6, 0.5, 252), "B": np.linspace(0.4, 0.5, 252)},
    )
    return {
        "returns": returns,
        "weights": weights,
        "metrics": {
            "total_return": 0.15,
            "annual_return": 0.12,
            "volatility": 0.18,
            "sharpe_ratio": 0.67,
            "max_drawdown": -0.08,
        },
    }


@pytest.fixture()
def output_dir(tmp_path):
    return tmp_path / "export_test"


# ---------------------------------------------------------------------------
# to_latex tests
# ---------------------------------------------------------------------------


class TestToLatex:
    def test_exports_pdf(self, mpl_fig, output_dir) -> None:
        path = to_latex(mpl_fig, output_dir / "fig.pdf")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_exports_png(self, mpl_fig, output_dir) -> None:
        path = to_latex(mpl_fig, output_dir / "fig.png")
        assert path.exists()
        assert path.suffix == ".png"

    def test_exports_svg(self, mpl_fig, output_dir) -> None:
        path = to_latex(mpl_fig, output_dir / "fig.svg")
        assert path.exists()
        content = path.read_text()
        assert "<svg" in content

    def test_creates_directory(self, mpl_fig, tmp_path) -> None:
        path = to_latex(mpl_fig, tmp_path / "deep" / "nested" / "fig.pdf")
        assert path.exists()

    def test_unsupported_format_raises(self, mpl_fig, output_dir) -> None:
        with pytest.raises(SPTInvariantError, match="Unsupported"):
            to_latex(mpl_fig, output_dir / "fig.xyz")

    def test_custom_dpi(self, mpl_fig, output_dir) -> None:
        path = to_latex(mpl_fig, output_dir / "fig.png", dpi=72)
        assert path.exists()


# ---------------------------------------------------------------------------
# to_pdf tests
# ---------------------------------------------------------------------------


class TestToPdf:
    def test_matplotlib_export(self, mpl_fig, output_dir) -> None:
        path = to_pdf(mpl_fig, output_dir / "fig.pdf")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_creates_directory(self, mpl_fig, tmp_path) -> None:
        path = to_pdf(mpl_fig, tmp_path / "new_dir" / "fig.pdf")
        assert path.exists()


# ---------------------------------------------------------------------------
# to_html tests
# ---------------------------------------------------------------------------


class TestToHtml:
    def test_matplotlib_svg_export(self, mpl_fig, output_dir) -> None:
        path = to_html(mpl_fig, output_dir / "fig.html")
        assert path.exists()
        content = path.read_text()
        assert "<svg" in content
        assert "<!DOCTYPE html>" in content

    def test_matplotlib_partial_html(self, mpl_fig, output_dir) -> None:
        path = to_html(mpl_fig, output_dir / "fig.html", full_html=False)
        assert path.exists()
        content = path.read_text()
        assert "<svg" in content
        assert "<html>" not in content

    def test_creates_directory(self, mpl_fig, tmp_path) -> None:
        path = to_html(mpl_fig, tmp_path / "sub" / "fig.html")
        assert path.exists()


# ---------------------------------------------------------------------------
# generate_report tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_html_report(self, backtest_result, output_dir) -> None:
        path = generate_report(backtest_result, output_dir / "report.html")
        assert path.exists()
        content = path.read_text()
        assert "Performance Metrics" in content
        assert "Cumulative Returns" in content
        assert "0.6700" in content  # sharpe ratio

    def test_html_with_title(self, backtest_result, output_dir) -> None:
        path = generate_report(
            backtest_result, output_dir / "report.html", title="My Strategy"
        )
        content = path.read_text()
        assert "My Strategy" in content

    def test_html_no_charts(self, backtest_result, output_dir) -> None:
        path = generate_report(
            backtest_result, output_dir / "report.html", include_charts=False
        )
        content = path.read_text()
        assert "Cumulative Returns" not in content

    def test_html_no_metrics(self, backtest_result, output_dir) -> None:
        path = generate_report(
            backtest_result, output_dir / "report.html", include_metrics=False
        )
        content = path.read_text()
        assert "Performance Metrics" not in content

    def test_html_no_weights(self, backtest_result, output_dir) -> None:
        path = generate_report(
            backtest_result, output_dir / "report.html", include_weights=False
        )
        content = path.read_text()
        assert "Final Portfolio Weights" not in content

    def test_pdf_report(self, backtest_result, output_dir) -> None:
        path = generate_report(backtest_result, output_dir / "report.pdf", format="pdf")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_latex_report(self, backtest_result, output_dir) -> None:
        path = generate_report(
            backtest_result, output_dir / "report.tex", format="latex"
        )
        assert path.exists()
        content = path.read_text()
        assert r"\documentclass" in content
        assert "Sharpe Ratio" in content
        assert r"\begin{tabular}" in content

    def test_invalid_format_raises(self, backtest_result, output_dir) -> None:
        with pytest.raises(SPTInvariantError, match="Unsupported format"):
            generate_report(backtest_result, output_dir / "x.docx", format="docx")

    def test_minimal_result(self, output_dir) -> None:
        """Report with only returns (no metrics or weights)."""
        result = {"returns": np.random.default_rng(0).normal(0, 0.01, 50)}
        path = generate_report(result, output_dir / "minimal.html")
        assert path.exists()

    def test_object_result(self, output_dir) -> None:
        """Test with an object that has attributes instead of dict."""

        class FakeResult:
            returns = np.random.default_rng(0).normal(0, 0.01, 50)
            metrics: dict[str, float] = {"sharpe_ratio": 1.5}  # noqa: RUF012

        path = generate_report(FakeResult(), output_dir / "obj.html")
        assert path.exists()
        content = path.read_text()
        assert "1.5000" in content

    def test_creates_parent_directories(self, backtest_result, tmp_path) -> None:
        path = generate_report(
            backtest_result, tmp_path / "a" / "b" / "c" / "report.html"
        )
        assert path.exists()
