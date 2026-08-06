"""Report export utilities for publication and sharing.

Provides export of figures and full backtest reports to LaTeX, PDF,
and self-contained HTML formats. Integrates with the visualization
module to produce publication-ready factsheets.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._preconditions import require

if TYPE_CHECKING:
    pass

__all__ = [
    "generate_report",
    "to_html",
    "to_latex",
    "to_pdf",
]


def to_latex(
    fig: Any,
    path: str | Path,
    width: str = r"\textwidth",
    dpi: int = 300,
) -> Path:
    r"""Export a matplotlib figure to LaTeX-compatible format (PGF/PDF).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to export.
    path : str or Path
        Output file path. Extension determines format:
        - ``.pgf``: PGF format (directly includeable in LaTeX).
        - ``.pdf``: PDF suitable for \includegraphics.
        - ``.eps``: Encapsulated PostScript.
    width : str
        LaTeX width specification for the figure wrapper.
    dpi : int
        Resolution for rasterized elements.

    Returns
    -------
    Path
        The path where the file was written.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix.lower()
    require(
        suffix in (".pgf", ".pdf", ".eps", ".png", ".svg"),
        f"Unsupported LaTeX export format: '{suffix}'. "
        "Use .pgf, .pdf, .eps, .png, or .svg.",
    )

    fig.savefig(
        str(output),
        dpi=dpi,
        bbox_inches="tight",
        backend="pgf" if suffix == ".pgf" else None,
    )
    plt.close(fig)

    return output


def to_pdf(
    fig: Any,
    path: str | Path,
    dpi: int = 300,
) -> Path:
    """Export a figure to PDF format.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure to export.
    path : str or Path
        Output file path (must end in .pdf).
    dpi : int
        Resolution for rasterized elements (matplotlib only).

    Returns
    -------
    Path
        The path where the file was written.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if _is_plotly_figure(fig):
        _require_plotly()
        fig.write_image(str(output), format="pdf")
    else:
        _require_matplotlib()
        import matplotlib.pyplot as plt

        fig.savefig(str(output), format="pdf", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return output


def to_html(
    fig: Any,
    path: str | Path,
    include_plotlyjs: bool | str = True,
    full_html: bool = True,
) -> Path:
    """Export a figure to self-contained HTML.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure or matplotlib.figure.Figure
        The figure to export. Plotly figures produce interactive HTML.
        Matplotlib figures are embedded as static SVG.
    path : str or Path
        Output file path.
    include_plotlyjs : bool or str
        For plotly: whether to include plotly.js in the file.
        ``True`` for self-contained, ``'cdn'`` for CDN link.
    full_html : bool
        Whether to produce a full HTML document or just a div.

    Returns
    -------
    Path
        The path where the file was written.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if _is_plotly_figure(fig):
        _require_plotly()
        fig.write_html(
            str(output),
            include_plotlyjs=include_plotlyjs,
            full_html=full_html,
        )
    else:
        _require_matplotlib()
        import io

        import matplotlib.pyplot as plt

        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg_content = buf.getvalue()

        if full_html:
            html_content = _wrap_svg_in_html(svg_content)
        else:
            html_content = svg_content

        output.write_text(html_content, encoding="utf-8")

    return output


def generate_report(
    backtest_result: Any,
    path: str | Path,
    format: str = "html",
    title: str = "Backtest Report",
    include_charts: bool = True,
    include_metrics: bool = True,
    include_weights: bool = True,
) -> Path:
    """Generate a full backtest factsheet report.

    Parameters
    ----------
    backtest_result : dict or object
        Backtest results containing at minimum:
        - ``returns``: Series or array of portfolio returns.
        - ``weights`` (optional): DataFrame of weight history.
        - ``metrics`` (optional): dict of performance metrics.
        - ``benchmark_returns`` (optional): Series of benchmark returns.
    path : str or Path
        Output file path.
    format : str
        Output format: ``'html'``, ``'pdf'``, or ``'latex'``.
    title : str
        Report title.
    include_charts : bool
        Whether to include performance charts.
    include_metrics : bool
        Whether to include a metrics summary table.
    include_weights : bool
        Whether to include weight evolution chart.

    Returns
    -------
    Path
        The path where the report was written.
    """
    require(
        format in ("html", "pdf", "latex"),
        f"Unsupported format '{format}'. Use 'html', 'pdf', or 'latex'.",
    )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    result_dict = _normalize_result(backtest_result)

    if format == "html":
        return _generate_html_report(
            result_dict, output, title, include_charts, include_metrics, include_weights
        )
    elif format == "pdf":
        return _generate_pdf_report(
            result_dict, output, title, include_charts, include_metrics, include_weights
        )
    else:
        return _generate_latex_report(result_dict, output, title, include_metrics)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for this export. "
            "Install with: pip install quantspt[viz]"
        ) from exc


def _require_plotly() -> None:
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "plotly is required for this export. "
            "Install with: pip install quantspt[viz]"
        ) from exc


def _is_plotly_figure(fig: Any) -> bool:
    try:
        import plotly.graph_objects as go

        return isinstance(fig, go.Figure)
    except ImportError:
        return False


def _wrap_svg_in_html(svg: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Figure</title></head>
<body style="display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;">
{svg}
</body>
</html>"""


def _normalize_result(result: Any) -> dict[str, Any]:
    """Convert various result types to a standard dict."""
    if isinstance(result, dict):
        return result

    out: dict[str, Any] = {}
    for attr in ("returns", "weights", "metrics", "benchmark_returns", "trades"):
        if hasattr(result, attr):
            out[attr] = getattr(result, attr)
    return out


def _generate_html_report(
    result: dict[str, Any],
    output: Path,
    title: str,
    include_charts: bool,
    include_metrics: bool,
    include_weights: bool,
) -> Path:
    """Generate an interactive HTML report."""

    sections: list[str] = []
    sections.append(f"<h1>{title}</h1>")

    if include_metrics and "metrics" in result:
        sections.append(_render_metrics_html(result["metrics"]))

    if include_charts and "returns" in result:
        sections.append(_render_returns_chart_html(result))

    if include_weights and "weights" in result:
        sections.append(_render_weights_html(result["weights"]))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 2rem; background: #fafafa; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 0.5rem; }}
h2 {{ color: #0f3460; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: white; text-align: left; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.chart {{ margin: 1.5rem 0; padding: 1rem; background: white;
          border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
{"".join(sections)}
<footer><p style="color:#888;font-size:0.8rem;margin-top:3rem;">
Generated by quantspt</p></footer>
</body>
</html>"""

    output.write_text(html, encoding="utf-8")
    return output


def _render_metrics_html(metrics: dict[str, Any]) -> str:
    rows = ""
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        label = key.replace("_", " ").title()
        rows += f"<tr><td style='text-align:left;font-weight:bold'>{label}</td><td>{formatted}</td></tr>\n"
    return f"<h2>Performance Metrics</h2>\n<table><tr><th>Metric</th><th>Value</th></tr>\n{rows}</table>"


def _render_returns_chart_html(result: dict[str, Any]) -> str:
    """Render cumulative returns as an inline SVG chart."""
    import numpy as np

    returns = result["returns"]
    if hasattr(returns, "values"):
        vals = returns.values.flatten()
    else:
        vals = np.asarray(returns).flatten()

    cumulative = np.cumprod(1.0 + vals)
    n = len(cumulative)

    width, height = 800, 300
    x_coords = [int(i / max(n - 1, 1) * (width - 60)) + 40 for i in range(n)]
    y_min, y_max = float(cumulative.min()), float(cumulative.max())
    y_range = max(y_max - y_min, 1e-10)
    y_coords = [
        int((1.0 - (v - y_min) / y_range) * (height - 40)) + 20 for v in cumulative
    ]

    points = " ".join(f"{x},{y}" for x, y in zip(x_coords, y_coords, strict=False))

    return f"""<div class="chart">
<h2>Cumulative Returns</h2>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
<polyline fill="none" stroke="#0f3460" stroke-width="2" points="{points}"/>
<text x="40" y="{height - 5}" font-size="10" fill="#666">t=0</text>
<text x="{width - 40}" y="{height - 5}" font-size="10" fill="#666">t={n}</text>
<text x="5" y="20" font-size="10" fill="#666">{y_max:.2f}</text>
<text x="5" y="{height - 25}" font-size="10" fill="#666">{y_min:.2f}</text>
</svg>
</div>"""


def _render_weights_html(weights: Any) -> str:
    """Render weight summary as HTML table."""

    if hasattr(weights, "columns"):
        tickers = list(weights.columns)
        final_weights = weights.iloc[-1].to_dict()
    elif isinstance(weights, dict):
        tickers = list(weights.keys())
        final_weights = weights
    else:
        return ""

    rows = ""
    for ticker in tickers:
        val = final_weights.get(ticker, 0.0)
        if isinstance(val, int | float):
            rows += f"<tr><td style='text-align:left'>{ticker}</td><td>{val:.4f}</td></tr>\n"
    return f"<h2>Final Portfolio Weights</h2>\n<table><tr><th>Asset</th><th>Weight</th></tr>\n{rows}</table>"


def _generate_pdf_report(
    result: dict[str, Any],
    output: Path,
    title: str,
    include_charts: bool,
    include_metrics: bool,
    include_weights: bool,
) -> Path:
    """Generate PDF report via matplotlib."""
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(str(output)) as pdf:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=18, weight="bold")
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        if include_metrics and "metrics" in result:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis("off")
            metrics = result["metrics"]
            table_data = [
                [
                    k.replace("_", " ").title(),
                    f"{v:.4f}" if isinstance(v, float) else str(v),
                ]
                for k, v in metrics.items()
            ]
            table = ax.table(
                cellText=table_data,
                colLabels=["Metric", "Value"],
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax.set_title("Performance Metrics", fontsize=14, pad=20)
            pdf.savefig(fig)
            plt.close(fig)

        if include_charts and "returns" in result:
            returns = result["returns"]
            if hasattr(returns, "values"):
                vals = returns.values.flatten()
            else:
                vals = np.asarray(returns).flatten()
            cumulative = np.cumprod(1.0 + vals)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(cumulative, color="#0f3460", linewidth=1.5)
            ax.set_title("Cumulative Returns", fontsize=14)
            ax.set_xlabel("Time")
            ax.set_ylabel("Growth of $1")
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig)
            plt.close(fig)

    return output


def _generate_latex_report(
    result: dict[str, Any],
    output: Path,
    title: str,
    include_metrics: bool,
) -> Path:
    """Generate LaTeX source for the report."""
    lines: list[str] = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\begin{document}",
        f"\\title{{{title}}}",
        r"\maketitle",
    ]

    if include_metrics and "metrics" in result:
        lines.append(r"\section{Performance Metrics}")
        lines.append(r"\begin{tabular}{lr}")
        lines.append(r"\toprule")
        lines.append(r"Metric & Value \\")
        lines.append(r"\midrule")
        for key, value in result["metrics"].items():
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                formatted = f"{value:.4f}"
            else:
                formatted = str(value)
            lines.append(f"{label} & {formatted} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    lines.append(r"\end{document}")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
