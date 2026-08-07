"""Performance benchmarks at realistic scale.

Measures wall-clock time for core SPT operations across five scenarios
(Small/Medium/Large/Stress/Massive) and three backends (numpy/numba/jax).

Run with:
    pytest tests/benchmarks/test_performance_benchmark.py -v -s --tb=short
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from scipy import stats

from quantspt._backends import (
    _reset_registry,
    get_backend,
    register_backend,
    set_backend,
)
from quantspt._backends.numpy_backend import NumpyBackend
from quantspt.backtesting import BacktestConfig, BacktestEngine
from quantspt.core.covariance import relative_covariance
from quantspt.core.generating_functions import DiversityGenerator
from quantspt.core.growth_rates import excess_growth_rate
from quantspt.core.processes import CorrelatedGBM, simulate_path
from quantspt.simulation import MonteCarloEngine

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"

# ---------------------------------------------------------------------------
# Data generation helpers (standalone, no fixtures needed for standalone run)
# ---------------------------------------------------------------------------


def _pareto_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    raw = (rng.pareto(1.0, size=n) + 1.0).astype(np.float64)
    return raw / raw.sum()


def _sector_covariance(rng: np.random.Generator, n: int) -> np.ndarray:
    n_sectors = min(n, 11)
    sector_sizes = np.diff(np.linspace(0, n, n_sectors + 1, dtype=int))
    vols = rng.uniform(0.15, 0.40, size=n)
    corr = np.full((n, n), 0.3)
    offset = 0
    for sz in sector_sizes:
        corr[offset : offset + sz, offset : offset + sz] = 0.6
        offset += sz
    np.fill_diagonal(corr, 1.0)
    D = np.diag(vols)
    cov = D @ corr @ D
    cov = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals[0] < 0:
        cov += np.eye(n) * (-eigvals[0] + 1e-8)
    return cov


def _realistic_returns(rng: np.random.Generator, n: int, n_days: int) -> np.ndarray:
    dt = 1.0 / 252.0
    daily_mean = 0.08 * dt
    vols = rng.uniform(0.15, 0.40, size=n)
    daily_std = vols * np.sqrt(dt)
    raw = stats.t.rvs(df=5.0, size=(n_days, n), random_state=rng)
    raw = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    return (daily_mean + daily_std * raw).astype(np.float64)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _time_fn(fn, *args, n_repeats: int = 3, **kwargs) -> tuple[float, Any]:
    """Time a function over n_repeats, return (median_seconds, last_result)."""
    times = []
    result = None
    for _ in range(n_repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


# ---------------------------------------------------------------------------
# PART A: Core math operations at scale
# ---------------------------------------------------------------------------

SCENARIO_PARAMS = [
    ("small", 10, 252),
    ("medium", 50, 1_260),
    ("large", 500, 2_520),
    ("massive", 2_000, 5_040),
]


@pytest.mark.slow
class TestCoreMathScaling:
    """Benchmark core SPT math across problem sizes."""

    _data: ClassVar[dict] = {}

    @classmethod
    @pytest.fixture(autouse=True, scope="class")
    def setup_data(cls):
        """Generate data once for the class."""
        cls._data = {}
        for name, n, n_days in SCENARIO_PARAMS:
            rng = np.random.default_rng(42)
            cls._data[name] = {
                "pi": _pareto_weights(rng, n),
                "cov": _sector_covariance(rng, n),
                "n": n,
                "n_days": n_days,
            }

    @pytest.mark.parametrize("scenario,n,n_days", SCENARIO_PARAMS)
    def test_excess_growth_rate_scaling(self, scenario, n, n_days):
        d = self._data[scenario]
        t, val = _time_fn(excess_growth_rate, d["pi"], d["cov"])
        print(f"\n  g* compute  n={n:>5d}  ->  {t * 1000:>10.3f} ms   val={val:.6e}")
        assert np.isfinite(val)
        assert val >= -1e-10  # non-negative for long-only

    @pytest.mark.parametrize("scenario,n,n_days", SCENARIO_PARAMS)
    def test_relative_covariance_scaling(self, scenario, n, n_days):
        d = self._data[scenario]
        t, tau = _time_fn(relative_covariance, d["cov"], d["pi"])
        print(
            f"\n  t^pi matrix  n={n:>5d}  ->  {t * 1000:>10.3f} ms   shape={tau.shape}"
        )
        assert tau.shape == (n, n)
        assert np.allclose(tau @ d["pi"], 0, atol=1e-8)

    @pytest.mark.parametrize("scenario,n,n_days", SCENARIO_PARAMS)
    def test_diversity_weights_scaling(self, scenario, n, n_days):
        d = self._data[scenario]
        gen = DiversityGenerator(p=0.5)
        t, w = _time_fn(gen.weights, d["pi"])
        print(
            f"\n  DivGen(0.5) n={n:>5d}  ->  {t * 1000:>10.3f} ms   sum={w.sum():.8f}"
        )
        assert abs(w.sum() - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# PART B: Backend comparison (numpy vs numba vs jax)
# ---------------------------------------------------------------------------


def _setup_numpy_backend():
    _reset_registry()
    register_backend("numpy", NumpyBackend())
    set_backend("numpy")
    return get_backend()


def _setup_numba_backend():
    from quantspt._backends.numba_backend import NumbaBackend

    _reset_registry()
    register_backend("numpy", NumpyBackend())
    nb = NumbaBackend()
    register_backend("numba", nb)
    set_backend("numba")
    return nb


def _setup_jax_backend():
    from quantspt._backends.jax_backend import JaxBackend

    _reset_registry()
    register_backend("numpy", NumpyBackend())
    jb = JaxBackend()
    register_backend("jax", jb)
    set_backend("jax")
    return jb


BACKEND_SIZES = [10, 50, 500, 2_000]


@pytest.mark.slow
class TestBackendComparison:
    """Compare numpy/numba/jax backend timings for core operations."""

    _data: ClassVar[dict] = {}

    @classmethod
    @pytest.fixture(autouse=True, scope="class")
    def setup_data(cls):
        cls._data = {}
        for n in BACKEND_SIZES:
            rng = np.random.default_rng(42)
            cls._data[n] = {
                "pi": _pareto_weights(rng, n),
                "cov": _sector_covariance(rng, n),
            }

    def test_excess_growth_rate_backends(self):
        results = {}
        for n in BACKEND_SIZES:
            d = self._data[n]
            pi, cov = d["pi"], d["cov"]
            row = {}

            # numpy
            be_np = _setup_numpy_backend()
            t_np, val_np = _time_fn(be_np.excess_growth_rate, pi, cov, n_repeats=5)
            row["numpy"] = t_np

            # numba (warmup first call, measure second onward)
            be_nb = _setup_numba_backend()
            be_nb.excess_growth_rate(pi, cov)  # JIT warmup
            t_nb, val_nb = _time_fn(be_nb.excess_growth_rate, pi, cov, n_repeats=5)
            row["numba"] = t_nb

            # jax (warmup first call)
            be_jax = _setup_jax_backend()
            be_jax.excess_growth_rate(pi, cov)  # JIT warmup
            t_jax, val_jax = _time_fn(be_jax.excess_growth_rate, pi, cov, n_repeats=5)
            row["jax"] = t_jax

            # verify correctness
            assert abs(val_np - val_nb) < 1e-6, f"numba mismatch at n={n}"
            assert abs(val_np - val_jax) < 1e-4, f"jax mismatch at n={n}"

            results[n] = row
            speedup_nb = t_np / t_nb if t_nb > 0 else float("inf")
            speedup_jax = t_np / t_jax if t_jax > 0 else float("inf")
            print(
                f"\n  g* n={n:>5d}:  numpy={t_np * 1000:.3f}ms  "
                f"numba={t_nb * 1000:.3f}ms ({speedup_nb:.1f}x)  "
                f"jax={t_jax * 1000:.3f}ms ({speedup_jax:.1f}x)"
            )

        _reset_registry()

    def test_relative_covariance_backends(self):
        results = {}
        for n in BACKEND_SIZES:
            d = self._data[n]
            pi, cov = d["pi"], d["cov"]
            row = {}

            be_np = _setup_numpy_backend()
            t_np, tau_np = _time_fn(be_np.relative_covariance, cov, pi, n_repeats=5)
            row["numpy"] = t_np

            be_nb = _setup_numba_backend()
            be_nb.relative_covariance(cov, pi)  # warmup
            t_nb, tau_nb = _time_fn(be_nb.relative_covariance, cov, pi, n_repeats=5)
            row["numba"] = t_nb

            be_jax = _setup_jax_backend()
            be_jax.relative_covariance(cov, pi)  # warmup
            t_jax, tau_jax = _time_fn(be_jax.relative_covariance, cov, pi, n_repeats=5)
            row["jax"] = t_jax

            assert np.allclose(tau_np, tau_nb, atol=1e-6), f"numba mismatch at n={n}"
            assert np.allclose(tau_np, tau_jax, atol=1e-4), f"jax mismatch at n={n}"

            results[n] = row
            speedup_nb = t_np / t_nb if t_nb > 0 else float("inf")
            speedup_jax = t_np / t_jax if t_jax > 0 else float("inf")
            print(
                f"\n  t^pi n={n:>5d}:  numpy={t_np * 1000:.3f}ms  "
                f"numba={t_nb * 1000:.3f}ms ({speedup_nb:.1f}x)  "
                f"jax={t_jax * 1000:.3f}ms ({speedup_jax:.1f}x)"
            )

        _reset_registry()

    def test_diversity_weights_backends(self):
        results = {}
        for n in BACKEND_SIZES:
            d = self._data[n]
            pi = d["pi"]
            row = {}

            be_np = _setup_numpy_backend()
            t_np, w_np = _time_fn(be_np.diversity_weights, pi, 0.5, n_repeats=5)
            row["numpy"] = t_np

            be_nb = _setup_numba_backend()
            t_nb, w_nb = _time_fn(be_nb.diversity_weights, pi, 0.5, n_repeats=5)
            row["numba"] = t_nb

            be_jax = _setup_jax_backend()
            be_jax.diversity_weights(pi, 0.5)  # warmup
            t_jax, w_jax = _time_fn(be_jax.diversity_weights, pi, 0.5, n_repeats=5)
            row["jax"] = t_jax

            assert np.allclose(w_np, w_nb, atol=1e-8)
            assert np.allclose(w_np, w_jax, atol=1e-4)

            results[n] = row
            print(
                f"\n  div_w n={n:>5d}:  numpy={t_np * 1000:.3f}ms  "
                f"numba={t_nb * 1000:.3f}ms  jax={t_jax * 1000:.3f}ms"
            )

        _reset_registry()


# ---------------------------------------------------------------------------
# PART C: Monte Carlo simulation at scale
# ---------------------------------------------------------------------------

MC_SCENARIOS = [
    ("small", 10, 252, 1_000),
    ("medium", 50, 252, 10_000),
    ("large", 500, 252, 1_000),
]


@pytest.mark.slow
class TestMonteCarloScaling:
    """Benchmark Monte Carlo simulation across problem sizes."""

    @pytest.mark.parametrize("name,n,n_steps,n_paths", MC_SCENARIOS)
    def test_simulate_single_path(self, name, n, n_steps, n_paths):
        rng = np.random.default_rng(42)
        mu = np.full(n, 0.08)
        cov = _sector_covariance(rng, n)
        x0 = np.ones(n) * 100.0
        process = CorrelatedGBM(mu=mu, cov=cov, x0=x0)

        # warmup
        simulate_path(process, T=1.0, n_steps=10, rng=np.random.default_rng(0))

        t, (_times_arr, path) = _time_fn(
            simulate_path,
            process,
            1.0,
            n_steps,
            np.random.default_rng(42),
            n_repeats=3,
        )
        print(f"\n  simulate_path n={n:>4d} steps={n_steps}  ->  {t:.4f}s")
        assert path.shape == (n_steps + 1, n)
        assert np.all(path > 0)

    @pytest.mark.parametrize("name,n,n_steps,n_paths", MC_SCENARIOS)
    def test_monte_carlo_engine(self, name, n, n_steps, n_paths):
        rng_gen = np.random.default_rng(42)
        mu = np.full(n, 0.08)
        cov = _sector_covariance(rng_gen, n)
        x0 = np.ones(n) * 100.0
        process = CorrelatedGBM(mu=mu, cov=cov, x0=x0)

        engine = MonteCarloEngine(
            process=process,
            n_paths=n_paths,
            T=1.0,
            n_steps=n_steps,
            seed=42,
        )

        t0 = time.perf_counter()
        result = engine.run()
        elapsed = time.perf_counter() - t0

        mc = result.data
        print(
            f"\n  MC engine n={n:>4d} paths={n_paths:>6d} steps={n_steps}"
            f"  ->  {elapsed:.2f}s  "
            f"(internal: {result.computation_time_ms:.0f}ms)"
        )
        assert mc.terminal_values.shape == (n_paths, n)
        assert np.all(np.isfinite(mc.mean))


# ---------------------------------------------------------------------------
# PART D: Backtest engine at scale
# ---------------------------------------------------------------------------

BACKTEST_SCENARIOS = [
    ("small_1y", 10, 252),
    ("medium_5y", 50, 1_260),
    ("large_10y", 500, 2_520),
]


@pytest.mark.slow
class TestBacktestScaling:
    """Benchmark BacktestEngine across problem sizes."""

    @pytest.mark.parametrize("name,n,n_days", BACKTEST_SCENARIOS)
    def test_backtest_diversity(self, name, n, n_days):
        rng = np.random.default_rng(42)
        pi0 = _pareto_weights(rng, n)
        raw_returns = _realistic_returns(rng, n, n_days)
        gross_returns = 1.0 + raw_returns

        gen = DiversityGenerator(p=0.5)
        engine = BacktestEngine(
            weight_func=gen.weights,
            returns=gross_returns,
            initial_weights=pi0,
            config=BacktestConfig(initial_value=1.0),
        )

        t0 = time.perf_counter()
        result = engine.run()
        elapsed = time.perf_counter() - t0

        bt = result.data
        log_rel = bt.log_relative_return()
        print(
            f"\n  Backtest n={n:>4d} days={n_days:>5d}  ->  {elapsed:.3f}s  "
            f"log(V^pi/V^μ)={log_rel:+.4f}  "
            f"rebalances={bt.n_rebalances}"
        )
        assert np.all(np.isfinite(bt.portfolio_values))
        assert bt.portfolio_values[-1] > 0


# ---------------------------------------------------------------------------
# PART E: Full benchmark suite runner (generates report)
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


@pytest.mark.slow
def test_generate_benchmark_report():
    """Run all benchmarks and write results to benchmarks/ directory."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}

    sizes = [10, 50, 500, 2_000]
    print("\n" + "=" * 72)
    print("PERFORMANCE BENCHMARK REPORT")
    print("=" * 72)

    # ----- Core math scaling (numpy) -----
    print("\n--- Core Math (numpy) ---")
    core_rows = []
    for n in sizes:
        rng_s = np.random.default_rng(42)
        pi = _pareto_weights(rng_s, n)
        cov = _sector_covariance(rng_s, n)

        t_egr, _ = _time_fn(excess_growth_rate, pi, cov, n_repeats=5)
        t_rcov, _ = _time_fn(relative_covariance, cov, pi, n_repeats=5)

        gen = DiversityGenerator(p=0.5)
        t_dw, _ = _time_fn(gen.weights, pi, n_repeats=5)

        core_rows.append({"n": n, "egr": t_egr, "rcov": t_rcov, "div_w": t_dw})
        print(
            f"  n={n:>5d}  g*={_fmt_time(t_egr):>8s}  "
            f"t^pi={_fmt_time(t_rcov):>8s}  "
            f"DivW={_fmt_time(t_dw):>8s}"
        )
    all_results["core_numpy"] = core_rows

    # ----- Backend comparison -----
    print("\n--- Backend Comparison: g* ---")
    backend_egr = []
    for n in sizes:
        rng_s = np.random.default_rng(42)
        pi = _pareto_weights(rng_s, n)
        cov = _sector_covariance(rng_s, n)
        row: dict[str, Any] = {"n": n}

        be_np = _setup_numpy_backend()
        t_np, _ = _time_fn(be_np.excess_growth_rate, pi, cov, n_repeats=5)
        row["numpy"] = t_np

        be_nb = _setup_numba_backend()
        be_nb.excess_growth_rate(pi, cov)  # warmup
        t_nb, _ = _time_fn(be_nb.excess_growth_rate, pi, cov, n_repeats=5)
        row["numba"] = t_nb

        be_jax = _setup_jax_backend()
        be_jax.excess_growth_rate(pi, cov)  # warmup
        t_jax, _ = _time_fn(be_jax.excess_growth_rate, pi, cov, n_repeats=5)
        row["jax"] = t_jax

        backend_egr.append(row)
        print(
            f"  n={n:>5d}  numpy={_fmt_time(t_np):>8s}  "
            f"numba={_fmt_time(t_nb):>8s} ({t_np / t_nb:.1f}x)  "
            f"jax={_fmt_time(t_jax):>8s} ({t_np / t_jax:.1f}x)"
        )

    print("\n--- Backend Comparison: t^pi ---")
    backend_rcov = []
    for n in sizes:
        rng_s = np.random.default_rng(42)
        pi = _pareto_weights(rng_s, n)
        cov = _sector_covariance(rng_s, n)
        row = {"n": n}

        be_np = _setup_numpy_backend()
        t_np, _ = _time_fn(be_np.relative_covariance, cov, pi, n_repeats=5)
        row["numpy"] = t_np

        be_nb = _setup_numba_backend()
        be_nb.relative_covariance(cov, pi)
        t_nb, _ = _time_fn(be_nb.relative_covariance, cov, pi, n_repeats=5)
        row["numba"] = t_nb

        be_jax = _setup_jax_backend()
        be_jax.relative_covariance(cov, pi)
        t_jax, _ = _time_fn(be_jax.relative_covariance, cov, pi, n_repeats=5)
        row["jax"] = t_jax

        backend_rcov.append(row)
        print(
            f"  n={n:>5d}  numpy={_fmt_time(t_np):>8s}  "
            f"numba={_fmt_time(t_nb):>8s} ({t_np / t_nb:.1f}x)  "
            f"jax={_fmt_time(t_jax):>8s} ({t_np / t_jax:.1f}x)"
        )

    _reset_registry()
    all_results["backend_egr"] = backend_egr
    all_results["backend_rcov"] = backend_rcov

    # ----- Simulation scaling -----
    print("\n--- Monte Carlo Simulation ---")
    mc_rows = []
    for n, n_paths in [(10, 1_000), (50, 1_000), (500, 100)]:
        rng_s = np.random.default_rng(42)
        mu = np.full(n, 0.08)
        cov = _sector_covariance(rng_s, n)
        x0 = np.ones(n) * 100.0
        process = CorrelatedGBM(mu=mu, cov=cov, x0=x0)

        engine = MonteCarloEngine(
            process=process,
            n_paths=n_paths,
            T=1.0,
            n_steps=252,
            seed=42,
        )
        t0 = time.perf_counter()
        result = engine.run()
        elapsed = time.perf_counter() - t0

        mc_rows.append({"n": n, "n_paths": n_paths, "time": elapsed})
        print(f"  MC n={n:>4d} paths={n_paths:>5d} steps=252  ->  {elapsed:.2f}s")
    all_results["monte_carlo"] = mc_rows

    # ----- Backtest scaling -----
    print("\n--- Backtest Engine ---")
    bt_rows = []
    for n, n_days in [(10, 252), (50, 1_260), (500, 2_520)]:
        rng_s = np.random.default_rng(42)
        pi0 = _pareto_weights(rng_s, n)
        raw_ret = _realistic_returns(rng_s, n, n_days)
        gross_ret = 1.0 + raw_ret

        gen = DiversityGenerator(p=0.5)
        engine = BacktestEngine(
            weight_func=gen.weights,
            returns=gross_ret,
            initial_weights=pi0,
            config=BacktestConfig(initial_value=1.0),
        )
        t0 = time.perf_counter()
        result = engine.run()
        elapsed = time.perf_counter() - t0
        bt = result.data
        bt_rows.append(
            {
                "n": n,
                "n_days": n_days,
                "time": elapsed,
                "log_rel": bt.log_relative_return(),
            }
        )
        print(
            f"  Backtest n={n:>4d} days={n_days:>5d}  ->  {elapsed:.3f}s  "
            f"log(V^pi/V^μ)={bt.log_relative_return():+.4f}"
        )
    all_results["backtest"] = bt_rows

    # ----- Write reports -----
    _write_performance_md(all_results)
    _write_backend_md(all_results)

    (BENCHMARKS_DIR / "raw_timings.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    print(f"\nReports saved to {BENCHMARKS_DIR}/")


def _write_performance_md(results: dict) -> None:
    lines = [
        "# Performance Benchmark Results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Core Math (NumPy backend)",
        "",
        "| Operation | n=10 | n=50 | n=500 | n=2000 |",
        "|-----------|------|------|-------|--------|",
    ]

    core = results["core_numpy"]
    for label, key in [
        ("g* compute", "egr"),
        ("t^pi matrix", "rcov"),
        ("DivGen weights", "div_w"),
    ]:
        vals = [_fmt_time(r[key]) for r in core]
        lines.append(f"| {label} | {' | '.join(vals)} |")

    lines += ["", "## Monte Carlo Simulation", ""]
    lines.append("| n_stocks | n_paths | n_steps | Time |")
    lines.append("|----------|---------|---------|------|")
    for r in results["monte_carlo"]:
        lines.append(f"| {r['n']} | {r['n_paths']} | 252 | {_fmt_time(r['time'])} |")

    lines += ["", "## Backtest Engine", ""]
    lines.append("| n_stocks | n_days | Time | log(V^pi/V^μ) |")
    lines.append("|----------|--------|------|-------------|")
    for r in results["backtest"]:
        lines.append(
            f"| {r['n']} | {r['n_days']} | {_fmt_time(r['time'])} | {r['log_rel']:+.4f} |"
        )

    (BENCHMARKS_DIR / "performance.md").write_text("\n".join(lines) + "\n")


def _write_backend_md(results: dict) -> None:
    lines = [
        "# Backend Comparison: NumPy vs Numba vs JAX",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "Numba and JAX timings exclude JIT warmup (first call).",
        "",
        "## Excess Growth Rate (g*)",
        "",
        "| n | NumPy | Numba | Speedup | JAX | Speedup |",
        "|---|-------|-------|---------|-----|---------|",
    ]

    for r in results["backend_egr"]:
        t_np, t_nb, t_jax = r["numpy"], r["numba"], r["jax"]
        lines.append(
            f"| {r['n']} | {_fmt_time(t_np)} | {_fmt_time(t_nb)} | "
            f"{t_np / t_nb:.1f}x | {_fmt_time(t_jax)} | {t_np / t_jax:.1f}x |"
        )

    lines += [
        "",
        "## Relative Covariance (t^pi)",
        "",
        "| n | NumPy | Numba | Speedup | JAX | Speedup |",
        "|---|-------|-------|---------|-----|---------|",
    ]

    for r in results["backend_rcov"]:
        t_np, t_nb, t_jax = r["numpy"], r["numba"], r["jax"]
        lines.append(
            f"| {r['n']} | {_fmt_time(t_np)} | {_fmt_time(t_nb)} | "
            f"{t_np / t_nb:.1f}x | {_fmt_time(t_jax)} | {t_np / t_jax:.1f}x |"
        )

    (BENCHMARKS_DIR / "backend_comparison.md").write_text("\n".join(lines) + "\n")
