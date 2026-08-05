# quantspt

**Stochastic Portfolio Theory for Python**

[![PyPI](https://img.shields.io/pypi/v/quantspt)](https://pypi.org/project/quantspt/)
[![Python](https://img.shields.io/pypi/pyversions/quantspt)](https://pypi.org/project/quantspt/)
[![Tests](https://github.com/XAheli/quantspt/actions/workflows/ci.yml/badge.svg)](https://github.com/XAheli/quantspt/actions)
[![codecov](https://codecov.io/gh/XAheli/quantspt/graph/badge.svg?token=7FYD87VZEW)](https://codecov.io/gh/XAheli/quantspt)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Sponsor](https://img.shields.io/badge/Sponsor-❤-ea4aaa?logo=github-sponsors)](https://github.com/sponsors/XAheli)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?logo=buy-me-a-coffee&logoColor=000000)](https://www.buymeacoffee.com/ahelipoddar)

---

quantspt is a rigorous implementation of **Stochastic Portfolio Theory** (SPT) as
developed by E. Robert Fernholz, Ioannis Karatzas, and collaborators. It provides
the complete mathematical apparatus for analysing portfolio behaviour through
diversification and volatility structure alone, without requiring return forecasts
or equilibrium assumptions.

Unlike classical mean-variance optimisation, SPT demonstrates that rebalancing
into a diversified portfolio generates a measurable **excess growth rate** --
a "diversification return" that can be computed directly from the covariance
structure of the market.

## Quick Install

```bash
pip install quantspt
```

## Hello World

```python
import numpy as np
from quantspt.core import excess_growth_rate, DiversityGenerator

# A 5-stock market: large-cap dominated weights
mu = np.array([0.40, 0.25, 0.20, 0.10, 0.05])

# Annualised covariance rate matrix (20-40% vol, 30% correlation)
sigma = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
rho = 0.3
a = rho * np.outer(sigma, sigma) + (1 - rho) * np.diag(sigma**2)

# Excess growth rate of the market portfolio
gamma_star_market = excess_growth_rate(mu, a)
print(f"Market excess growth rate:  {gamma_star_market:.4f}")

# Diversity-weighted portfolio (Fernholz, p=0.76)
gen = DiversityGenerator(p=0.76)
pi = gen.weights(mu)
gamma_star_div = excess_growth_rate(pi, a)
print(f"Diversity portfolio excess: {gamma_star_div:.4f}")
print(f"Diversification gain:       {gamma_star_div - gamma_star_market:.4f}")
print(f"Portfolio weights:           {np.round(pi, 4)}")
```

```
Market excess growth rate:  0.0196
Diversity portfolio excess: 0.0214
Diversification gain:       0.0018
Portfolio weights:           [0.3515 0.2459 0.2076 0.1226 0.0724]
```

The diversity-weighted portfolio earns 18 basis points per year of additional
growth purely from its more balanced allocation -- no alpha forecast required.

## Key Features

- **Excess growth rate computation** -- the fundamental quantity of SPT,
  measuring diversification return from the covariance structure alone
- **Functionally Generated Portfolios (FGPs)** -- mechanically produce portfolio
  weights from generating functions with known performance decompositions
- **Master formula verification** -- decompose FGP performance into boundary
  and drift terms; verify the theorem on simulated or empirical data
- **Diversity and arbitrage conditions** -- test whether a market satisfies
  weak or strict diversity, and compute minimum horizons for relative arbitrage
- **Generating function library** -- diversity-weighted, entropy-weighted,
  modified entropy, inverse volatility, and custom user-defined generators
- **Market weight dynamics** -- ranked weights, capital distribution curves,
  coherence verification, and weight diffusion analysis
- **Atlas model** -- closed-form results for the Banner-Fernholz-Karatzas Atlas
  model of equity markets
- **SDE simulation engine** -- Euler-Maruyama, Milstein, and exact GBM
  discretisation for Monte Carlo studies
- **Relative covariance theory** -- full implementation of the relative
  covariance matrix and its spectral properties

## Architecture

```
quantspt/
|
|-- core/                 Pure mathematical definitions (no estimation, no side effects)
|   |-- growth_rates      Excess growth rate, portfolio growth rate, bounds
|   |-- generating_functions   FGP framework: diversity, entropy, custom generators
|   |-- master_formula    Master formula decomposition and verification
|   |-- covariance        Relative covariance tau^pi, non-degeneracy conditions
|   |-- diversity         p-diversity, entropy, HHI, arbitrage horizons
|   |-- market            Market weights, ranked weights, capital distribution
|   |-- portfolio         Value process, relative returns, turnover
|   +-- processes         SDE discretisation schemes
|
|-- estimation/           Statistical estimation from price data
|-- models/               Market models (Atlas, GBM, volatility-stabilised)
|-- rank/                 Rank-based dynamics and capital distribution theory
|-- arbitrage/            Relative arbitrage detection and construction
|-- optimization/         SPT-native portfolio optimisation (cvxpy)
|-- simulation/           Monte Carlo simulation engine (numba-accelerated)
|-- backtesting/          Historical backtesting with SPT attribution
|-- data/                 Pluggable data providers (yfinance, custom)
+-- visualization/        Publication-quality plots (matplotlib, plotly)
```

## Documentation

Full documentation is hosted at [quantspt.readthedocs.io](https://quantspt.readthedocs.io).

## Mathematical Foundation

quantspt implements the mathematical framework developed in:

- **Fernholz, E.R.** (2002). *Stochastic Portfolio Theory*. Springer.
  The foundational monograph defining the field.

- **Fernholz, E.R. & Karatzas, I.** (2009). "Stochastic Portfolio Theory:
  A Survey." In *Handbook of Numerical Analysis*, Vol. XV, pp. 89-167.
  Comprehensive treatment of excess growth rates, FGPs, and the master formula.

- **Banner, A., Fernholz, E.R. & Karatzas, I.** (2005). "Atlas Models of
  Equity Markets." *Annals of Applied Probability*, 15(4), pp. 2296-2330.
  Rank-based models with explicit growth rate computations.

- **Fernholz, E.R., Karatzas, I. & Kardaras, C.** (2005). "Diversity and
  Relative Arbitrage in Equity Markets." *Finance and Stochastics*, 9(1),
  pp. 1-27. Conditions for model-free outperformance.

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
| data | yfinance, pydantic | `pip install "quantspt[data]"` |
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
