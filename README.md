# quantspt

**Stochastic Portfolio Theory for Python**

[![PyPI](https://img.shields.io/pypi/v/quantspt)](https://pypi.org/project/quantspt/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://pypi.org/project/quantspt/)
[![CI](https://github.com/XAheli/quantspt/actions/workflows/ci.yml/badge.svg)](https://github.com/XAheli/quantspt/actions)
[![codecov](https://codecov.io/gh/XAheli/quantspt/graph/badge.svg?token=7FYD87VZEW)](https://codecov.io/gh/XAheli/quantspt)
[![MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Sponsor](https://img.shields.io/badge/sponsor-❤-ea4aaa?logo=github-sponsors)](https://github.com/sponsors/XAheli)
[![Buy Me A Coffee](https://img.shields.io/badge/buy%20me%20a%20coffee-ffdd00?logo=buy-me-a-coffee&logoColor=000000)](https://www.buymeacoffee.com/ahelipoddar)

---

quantspt is a rigorous implementation of **Stochastic Portfolio Theory** (SPT) as
developed by E. Robert Fernholz, Ioannis Karatzas, and collaborators. It provides
the complete mathematical apparatus for analysing portfolio behaviour through
diversification and volatility structure alone, without requiring return forecasts
or equilibrium assumptions.

Classical diversity strategies (the `DiversityGenerator` family) underperform in
concentrating markets because they carry a **boundary term** that increases with
market concentration. **`GammaGradientStrategy`** addresses this by targeting the
excess growth rate gradient directly — capturing the drift (diversification return)
without boundary exposure.

## Quick Install

```bash
pip install quantspt
```

## Hello World

```python
import numpy as np
from quantspt import GammaGradientStrategy

mu = np.array([0.40, 0.25, 0.20, 0.10, 0.05])
cov = np.diag([0.04, 0.06, 0.08, 0.10, 0.12])

strategy = GammaGradientStrategy(lambda_scale=0.1, max_weight=0.05)
weights = strategy.compute_weights(mu, cov)
print(f"Weights: {np.round(weights, 4)}")
```

On S&P 500 data (2020-2026), `GammaGradientStrategy` delivered **+269 bps/yr
beta-adjusted alpha** with zero size-factor correlation and low turnover (~2x/year).

## Why GammaGradientStrategy?

The Fernholz master formula decomposes FGP outperformance into two terms:

- **Drift** (positive) — the diversification return, proportional to γ\*
- **Boundary** (can be negative) — penalty from market concentration

Classical diversity-weighted portfolios (`DiversityGenerator(p)`) earn both.
In concentrating markets the boundary term dominates, eroding returns.

`GammaGradientStrategy` targets γ\* directly through its gradient:

```
w_i = μ_i + λ · ∂γ*/∂π_i |_{π=μ}
```

No generating function means no boundary term. The strategy captures drift
without structural exposure to concentration risk.

## Key Features

- **GammaGradientStrategy** — direct γ\* gradient targeting, the recommended
  strategy for capturing the volatility harvesting premium
- **Excess growth rate computation** — the fundamental quantity of SPT,
  measuring diversification return from the covariance structure alone
- **Functionally Generated Portfolios (FGPs)** — diversity-weighted, entropy-weighted,
  modified entropy, and custom user-defined generators with known performance
  decompositions
- **SPT Universe Selection** — `SPTUniverseSelector` picks stocks that maximise
  the excess growth rate by scoring idiosyncratic volatility, pairwise correlation,
  and boundary risk
- **Master formula verification** — decompose FGP performance into boundary
  and drift terms; verify the theorem on simulated or empirical data
- **Market models** — Atlas, correlated GBM, and volatility-stabilised markets
  with closed-form results
- **SDE simulation engine** — Euler-Maruyama, Milstein, and exact GBM
  discretisation for Monte Carlo studies
- **ML extensions** — composable training losses, regime detection, and
  covariance estimation (factor models, RMT denoising)
- **Integrations** — adapters for vectorbt and backtrader

## Architecture

```
quantspt/
│
├── strategies/           Direct optimization strategies
│   └── gamma_gradient    GammaGradientStrategy (+269 bps alpha)
│
├── core/                 Pure mathematical definitions
│   ├── growth_rates      Excess growth rate, portfolio growth rate, bounds
│   ├── generating_functions   FGP framework: diversity, entropy, custom
│   ├── master_formula    Master formula decomposition and verification
│   ├── covariance        Relative covariance, non-degeneracy conditions
│   ├── diversity         p-diversity, entropy, HHI, arbitrage horizons
│   └── processes         SDE discretisation schemes
│
├── universe/             SPT-optimised stock selection
│   ├── selector          SPTUniverseSelector (composite scoring)
│   ├── criteria          γ* contribution, boundary risk, idiosyncratic vol
│   └── reconstitution    Hysteresis-aware universe rebalancing
│
├── estimation/           Statistical estimation from price data
├── models/               Market models (Atlas, GBM, volatility-stabilised)
├── optimization/         SPT-native portfolio optimisation (cvxpy)
├── simulation/           Monte Carlo simulation engine (numba-accelerated)
├── backtesting/          Historical backtesting with SPT attribution
├── ml/                   Losses, regime detection, covariance estimators
├── causal/               Causal structure learning and interventional cov
├── integrations/         vectorbt and backtrader adapters
│
└── experimental/         Research-stage strategies
    ├── neural_fgp        ICNN-based learned generating functions
    ├── adaptive_fgp      Correction-anchored adaptive generators
    └── conditional_fgp   Covariance-conditional generating functions
```

## Mathematical Foundation

The FGP framework (Fernholz 2002) provides the mathematical foundation.
The diversity-weighted generator `G_p(μ) = (Σ μ_i^p)^{1/p}` produces
portfolios with a known performance decomposition via the master formula.

Research on S&P 500 data (2020-2026) revealed that in concentrating markets,
the boundary term of classical FGPs can dominate the drift term.
`GammaGradientStrategy` was developed to capture drift directly:

```
γ*(π) = ½[Σ_i π_i a_{ii} − π'aπ]

∂γ*/∂π_i = ½[a_{ii} − 2·(aπ)_i]
```

This targets stocks with high idiosyncratic variance relative to portfolio
covariance — the mathematical source of the rebalancing premium.

## References

- **Fernholz, E.R.** (2002). *Stochastic Portfolio Theory*. Springer.
- **Fernholz, E.R. & Karatzas, I.** (2009). "Stochastic Portfolio Theory:
  A Survey." In *Handbook of Numerical Analysis*, Vol. XV, pp. 89-167.
- **Banner, A., Fernholz, E.R. & Karatzas, I.** (2005). "Atlas Models of
  Equity Markets." *Annals of Applied Probability*, 15(4), pp. 2296-2330.
- **Fernholz, E.R., Karatzas, I. & Kardaras, C.** (2005). "Diversity and
  Relative Arbitrage in Equity Markets." *Finance and Stochastics*, 9(1),
  pp. 1-27.

## Installation

### Core (NumPy, Pandas, SciPy only)

```bash
pip install quantspt
```

### With optional dependencies

| Extra | Includes | Command |
|-------|----------|---------|
| viz | matplotlib, plotly, seaborn | `pip install "quantspt[viz]"` |
| opt | cvxpy | `pip install "quantspt[opt]"` |
| sim | numba | `pip install "quantspt[sim]"` |
| gpu | jax, jaxlib | `pip install "quantspt[gpu]"` |
| data | pydantic (CSV/Parquet providers) | `pip install "quantspt[data]"` |
| ml | torch, optuna, hmmlearn, ruptures | `pip install "quantspt[ml]"` |
| causal | pgmpy, networkx | `pip install "quantspt[causal]"` |
| dev | pytest, mypy, ruff, pre-commit | `pip install "quantspt[dev]"` |
| all | everything above | `pip install "quantspt[all]"` |

### From source

```bash
git clone https://github.com/XAheli/quantspt.git
cd quantspt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on development setup, testing, code style, and the pull request process.

## Citation

If you use quantspt in academic work, please cite:

```bibtex
@software{poddar2026quantspt,
  author       = {Poddar, Aheli},
  title        = {quantspt: Stochastic Portfolio Theory for Python},
  year         = {2026},
  url          = {https://github.com/XAheli/quantspt},
  version      = {0.1.0},
}
```

## License

MIT -- see [LICENSE](LICENSE).
