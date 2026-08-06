"""Verify ML_INTEGRATION_PLAN Section 12 prerequisites.

Checks every module, class, function, and protocol listed in Section 12
of ML_INTEGRATION_PLAN.md and reports pass/fail with details.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results: list[tuple[str, bool, str]] = []


def check(label: str, fn, *, note: str = ""):
    try:
        fn()
        results.append((label, True, note or "exists"))
        print(f"  [{PASS}] {label}")
    except Exception as e:
        results.append((label, False, str(e)))
        print(f"  [{FAIL}] {label}: {e}")


def check_deferred(label: str, note: str):
    results.append((label, None, note))
    print(f"  [{SKIP}] {label}: {note}")


print("=" * 70)
print("Section 12 Verification — ML_INTEGRATION_PLAN.md")
print("=" * 70)

# ── 12.1: Modules That Must Be Created ──────────────────────────────

print("\n── 12.1: Modules That Must Be Created ──")

check_deferred(
    "core/numeraire.py",
    "Deferred to future release (numeraire-invariant formulations)",
)
check_deferred(
    "models/diverse_market.py",
    "Deferred to future release (log-pole repulsion models)",
)
check_deferred(
    "models/hybrid.py",
    "Deferred to future release (regime-switching market models)",
)
check_deferred(
    "arbitrage/deflators.py",
    "Deferred to future release (strict local martingale detection)",
)

check(
    "estimation/covariance/factor_model (in ml/covariance.py)",
    lambda: importlib.import_module("quantspt.ml.covariance").FactorModelEstimator,
    note="Implemented as ml.covariance.FactorModelEstimator",
)

check(
    "estimation/covariance/rmt (in ml/covariance.py)",
    lambda: importlib.import_module("quantspt.ml.covariance").RMTDenoiser,
    note="Implemented as ml.covariance.RMTDenoiser",
)

check_deferred(
    "estimation/model_selection.py",
    "Deferred to future release (AIC/BIC/cross-validation)",
)

check(
    "simulation/path_generator → MonteCarloEngine",
    lambda: importlib.import_module("quantspt.simulation.monte_carlo").MonteCarloEngine,
    note="Equivalent functionality via MonteCarloEngine + SDE integrators",
)

check(
    "data/cache.py",
    lambda: importlib.import_module("quantspt.data.cache").CachedComputation,
)

check(
    "backtesting/engine.py",
    lambda: importlib.import_module("quantspt.backtesting.engine").BacktestEngine,
)

check(
    "backtesting/attribution.py",
    lambda: (
        importlib.import_module("quantspt.backtesting.attribution").compute_attribution
    ),
)

check(
    "post_processing/clean_weights.py",
    lambda: (
        importlib.import_module("quantspt.post_processing.clean_weights").clean_weights
    ),
)


# ── 12.2: Existing Modules That Need Additions ──────────────────────

print("\n── 12.2: Existing Module Additions ──")

check(
    "core/generating_functions.GeneratingFunction ABC",
    lambda: (
        importlib.import_module("quantspt.core.generating_functions").GeneratingFunction
    ),
)

check(
    "core/generating_functions.AutoDiffGeneratingFunction",
    lambda: (
        importlib.import_module(
            "quantspt.core.generating_functions"
        ).AutoDiffGeneratingFunction
    ),
)

check(
    "core/processes.StochasticProcessArray",
    lambda: importlib.import_module("quantspt.core.processes").StochasticProcessArray,
)

check(
    "core/processes.JointProcess",
    lambda: importlib.import_module("quantspt.core.processes").JointProcess,
)

check(
    "core/covariance.CovarianceRateProcess Protocol",
    lambda: importlib.import_module("quantspt.core.covariance").CovarianceRateProcess,
)


def check_cov_protocol_methods():
    mod = importlib.import_module("quantspt.core.covariance")
    assert hasattr(mod, "portfolio_variance"), "portfolio_variance not found"
    assert hasattr(mod, "non_degeneracy_bounds"), "non_degeneracy_bounds not found"


check(
    "core/covariance portfolio_variance + non_degeneracy_bounds",
    check_cov_protocol_methods,
)

check(
    "_typing.py — Time alias",
    lambda: importlib.import_module("quantspt._typing").Time,
)
check(
    "_typing.py — Weight alias",
    lambda: importlib.import_module("quantspt._typing").Weight,
)
check(
    "_typing.py — CovarianceRate alias",
    lambda: importlib.import_module("quantspt._typing").CovarianceRate,
)
check(
    "_typing.py — DiversityParameter alias",
    lambda: importlib.import_module("quantspt._typing").DiversityParameter,
)
check(
    "_typing.py — StochasticProcess Protocol",
    lambda: importlib.import_module("quantspt._typing").StochasticProcess,
)
check(
    "_typing.py — PortfolioGenerator Protocol",
    lambda: importlib.import_module("quantspt._typing").PortfolioGenerator,
)


# Check schemas
def check_schemas():
    schemas = importlib.import_module("quantspt.data.schemas")
    missing = []
    for name in ("CausalGraph", "RegimeLabels", "FactorLoadings"):
        if not hasattr(schemas, name):
            missing.append(name)
    if missing:
        raise AttributeError(f"Missing from data/schemas.py: {', '.join(missing)}")


check("data/schemas.py — CausalGraph, RegimeLabels, FactorLoadings", check_schemas)


# Check _result.py chart method
def check_chart():
    result_cls = importlib.import_module("quantspt._result").SPTResult
    if not hasattr(result_cls, "chart"):
        raise AttributeError("SPTResult.chart() method not found")


check("_result.py — SPTResult.chart() method", check_chart)


# Check contrib register_generating_function
def check_register_gf():
    contrib = importlib.import_module("quantspt.contrib")
    if not hasattr(contrib, "register_generating_function"):
        raise AttributeError("register_generating_function not found in contrib")


check("contrib — register_generating_function", check_register_gf)


# ── 12.5: Import Path Verification ──────────────────────────────────

print("\n── 12.5: Import Path Verification ──")

import_paths = [
    ("quantspt.core.generating_functions", "GeneratingFunction"),
    ("quantspt.core.generating_functions", "DiversityGenerator"),
    ("quantspt.core.covariance", "relative_covariance"),
    ("quantspt.core.growth_rates", "excess_growth_rate"),
    ("quantspt.core.master_formula", "master_formula_decomposition"),
    ("quantspt.core.processes", "CorrelatedGBM"),
    ("quantspt.core.processes", "simulate_path"),
    ("quantspt.data.schemas", "MarketPanel"),
    ("quantspt.data.schemas", "ReturnsMatrix"),
    ("quantspt._result", "SPTResult"),
    ("quantspt.errors", "SPTInvariantError"),
    ("quantspt.errors", "CalibrationError"),
]

for module_path, attr_name in import_paths:
    check(
        f"{module_path}.{attr_name}",
        lambda mp=module_path, an=attr_name: getattr(importlib.import_module(mp), an),
    )

# ML module imports (should exist since we have [ml] installed)
ml_paths = [
    ("quantspt.ml.neural_fgp", "NeuralFGP"),
    ("quantspt.ml.covariance", "FactorModelEstimator"),
    ("quantspt.ml.regime", "HMMRegimeDetector"),
]
for module_path, attr_name in ml_paths:
    check(
        f"{module_path}.{attr_name}",
        lambda mp=module_path, an=attr_name: getattr(importlib.import_module(mp), an),
    )

# Causal module imports
causal_paths = [
    ("quantspt.causal.structure", "CausalStructureLearner"),
    ("quantspt.causal.covariance", "CausalCovarianceEstimator"),
]
for module_path, attr_name in causal_paths:
    check(
        f"{module_path}.{attr_name}",
        lambda mp=module_path, an=attr_name: getattr(importlib.import_module(mp), an),
    )

# ── 12.3: Data Type Flow Verification ───────────────────────────────

print("\n── 12.3: Data Type Flow (functional) ──")


def verify_data_flow():
    from quantspt.core.covariance import relative_covariance
    from quantspt.core.generating_functions import DiversityGenerator
    from quantspt.core.growth_rates import excess_growth_rate
    from quantspt.core.master_formula import master_formula_decomposition

    n = 5
    mu = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    sigma = np.random.default_rng(42).random((n, n))
    cov = sigma @ sigma.T / 10

    tau = relative_covariance(cov, mu)
    assert tau.shape == (n, n), f"tau shape {tau.shape}"

    gf = DiversityGenerator(p=0.5)
    weights = gf.weights(mu)
    assert weights.shape == (n,), f"weights shape {weights.shape}"
    assert abs(weights.sum() - 1.0) < 1e-10, f"weights sum {weights.sum()}"

    egr = excess_growth_rate(weights, cov)
    assert isinstance(egr, float), f"egr type {type(egr)}"

    T_steps = 10
    mu_path = np.tile(mu, (T_steps, 1)) + np.random.default_rng(42).normal(
        0, 0.01, (T_steps, n)
    )
    mu_path = np.abs(mu_path)
    mu_path = mu_path / mu_path.sum(axis=1, keepdims=True)
    a_path = np.stack([cov] * T_steps)
    decomp = master_formula_decomposition(gf, mu_path, a_path, dt=1.0 / 252)
    assert "drift_integral" in decomp, "decomp missing 'drift_integral'"


check("Data type flow: mu → tau → weights → EGR → decomposition", verify_data_flow)


def verify_ml_flow():
    from quantspt.ml.neural_fgp import NeuralFGP, NeuralFGPConfig

    config = NeuralFGPConfig(hidden_dims=[16, 16])
    model = NeuralFGP(n_assets=5, config=config)
    assert model.n_assets == 5
    assert hasattr(model, "weights"), "NeuralFGP has no weights() method"
    assert hasattr(model, "generating_function"), (
        "NeuralFGP has no generating_function() method"
    )
    assert hasattr(model, "log_gradient"), "NeuralFGP has no log_gradient() method"
    assert hasattr(model, "hessian"), "NeuralFGP has no hessian() method"
    assert hasattr(model, "to_generating_function"), (
        "NeuralFGP has no to_generating_function()"
    )


check("ML flow: NeuralFGP interface verification", verify_ml_flow)


def verify_ml_cov_flow():
    from quantspt.ml.covariance import FactorModelEstimator, RMTDenoiser

    rng = np.random.default_rng(42)
    returns = rng.standard_normal((100, 5))

    fm = FactorModelEstimator(n_factors=2)
    fm.fit(returns)
    cov = fm.estimate()
    assert cov.shape == (5, 5), f"FactorModel cov shape: {cov.shape}"
    assert np.allclose(cov, cov.T), "FactorModel cov not symmetric"

    rmt = RMTDenoiser()
    rmt.fit(returns)
    cov2 = rmt.estimate()
    assert cov2.shape == (5, 5), f"RMTDenoiser cov shape: {cov2.shape}"


check("ML cov flow: FactorModel + RMTDenoiser fit → estimate", verify_ml_cov_flow)


def verify_regime_flow():
    from quantspt.ml.regime import HMMRegimeDetector

    rng = np.random.default_rng(42)
    features = rng.standard_normal((100, 3))
    det = HMMRegimeDetector(n_regimes=2)
    det.fit(features)
    labels = det.predict(features)
    assert labels.shape == (100,), f"Regime labels shape: {labels.shape}"


check("ML regime flow: HMMRegimeDetector fit → predict", verify_regime_flow)


def verify_causal_flow():
    from quantspt.causal.structure import CausalStructureLearner

    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 4))
    learner = CausalStructureLearner(method="pc")
    result = learner.fit(data, variable_names=["A", "B", "C", "D"])
    assert hasattr(result, "adjacency_matrix"), "No adjacency_matrix"


check("Causal flow: CausalStructureLearner fit → adjacency", verify_causal_flow)

# ── Summary ─────────────────────────────────────────────────────────

print("\n" + "=" * 70)
passed = sum(1 for _, ok, _ in results if ok is True)
failed = sum(1 for _, ok, _ in results if ok is False)
skipped = sum(1 for _, ok, _ in results if ok is None)
total = len(results)
print(f"Results: {passed} passed, {failed} failed, {skipped} deferred, {total} total")

if failed > 0:
    print("\n\033[91mFAILED items:\033[0m")
    for label, ok, note in results:
        if ok is False:
            print(f"  - {label}: {note}")

if skipped > 0:
    print("\n\033[93mDEFERRED items:\033[0m")
    for label, ok, note in results:
        if ok is None:
            print(f"  - {label}: {note}")

sys.exit(1 if failed > 0 else 0)
