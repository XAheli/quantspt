# quantspt

**The Definitive Python Library for Stochastic Portfolio Theory**

[![PyPI version](https://img.shields.io/pypi/v/quantspt)](https://pypi.org/project/quantspt/)
[![Python](https://img.shields.io/pypi/pyversions/quantspt)](https://pypi.org/project/quantspt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/XAheli/quantspt/actions/workflows/ci.yml/badge.svg)](https://github.com/XAheli/quantspt/actions)

`quantspt` implements the complete mathematical apparatus of **Stochastic Portfolio Theory** (SPT) as developed by E. Robert Fernholz, Ioannis Karatzas, and collaborators — the first and only production-quality SPT library.

## What is Stochastic Portfolio Theory?

SPT is a mathematical framework for portfolio analysis that, unlike Modern Portfolio Theory (MPT):

- **Requires no return forecasts** — performance comes from diversification and rebalancing
- **Provides model-free relative arbitrage** — diversity-weighted portfolios provably outperform the market over sufficiently long horizons
- **Explains why rebalancing creates value** — the *excess growth rate* γ\* quantifies the "diversification return"

## Quick Start

```python
import quantspt as spt

# Check version
print(spt.__version__)  # 0.1.0.dev0
```

## Installation

### From source (development)

```bash
git clone https://github.com/XAheli/quantspt.git
cd quantspt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Extras

| Extra | Packages | Install |
|-------|----------|---------|
| Core | numpy, pandas, scipy | `pip install quantspt` |
| Visualization | + matplotlib, plotly, seaborn | `pip install quantspt[viz]` |
| Optimization | + cvxpy | `pip install quantspt[opt]` |
| Simulation | + numba | `pip install quantspt[sim]` |
| GPU | + jax, jaxlib | `pip install quantspt[gpu]` |
| Data | + yfinance, pydantic | `pip install quantspt[data]` |
| Development | + pytest, mypy, ruff, … | `pip install quantspt[dev]` |
| Everything | all of the above | `pip install quantspt[all]` |

## Package Architecture

```
quantspt/
├── core/              # Pure mathematical definitions (NO estimation)
├── models/            # Market models (Atlas, GBM, volatility-stabilised, …)
├── rank/              # Rank-based theory and dynamics
├── arbitrage/         # Relative arbitrage detection and construction
├── estimation/        # Statistical estimation from data
├── optimization/      # Portfolio optimisation (correct SPT formulation)
├── simulation/        # Monte Carlo & SDE simulation engine
├── backtesting/       # Historical backtesting with SPT attribution
├── data/              # Data layer with pluggable providers
├── visualization/     # Publication-quality plots
├── contrib/           # Plugin system and community extensions
└── utils/             # Shared numerical utilities
```

## Key Concepts

### Excess Growth Rate (γ\*)

The fundamental quantity of SPT — the "diversification return" earned purely from rebalancing:

$$\gamma^*_\pi(t) = \frac{1}{2}\left[\sum_i \pi_i(t) a_{ii}(t) - \sum_{i,j} \pi_i(t) a_{ij}(t) \pi_j(t)\right]$$

### Functionally Generated Portfolios (FGPs)

A C² positive function G on the simplex mechanically produces portfolio weights with a known performance decomposition via the **Master Formula**:

$$\log\frac{V^\pi(T)}{V^\mu(T)} = \log\frac{G(\mu(T))}{G(\mu(0))} + \int_0^T g(t)\,dt$$

### Relative Arbitrage

In weakly diverse markets, diversity-weighted portfolios outperform the market a.s. for horizons T ≥ T\* = 2 log(n) / (p ε δ).

## Mathematical References

- **F&K Survey (2008)**: Fernholz & Karatzas, "Stochastic Portfolio Theory: A Survey"
- **BFK Atlas (2005)**: Banner, Fernholz & Karatzas, "Atlas Models of Equity Markets"
- **FKK Diversity (2005)**: Fernholz, Karatzas & Kardaras, "Diversity and Relative Arbitrage"
- **Lukacs Lectures (2006)**: Karatzas, "Lectures on the Mathematics of Finance"
- **Fernholz (2002)**: *Stochastic Portfolio Theory* (Springer monograph)

## Development

```bash
# Run tests
pytest tests/

# Type checking
mypy quantspt/ --strict

# Linting and formatting
ruff check quantspt/ tests/
ruff format quantspt/ tests/
```

## License

MIT — see [LICENSE](LICENSE).

## Author

**Aheli Poddar** — [ahelipoddar2003@gmail.com](mailto:ahelipoddar2003@gmail.com)
