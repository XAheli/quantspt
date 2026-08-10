# γ* Gradient Targeting Strategy — Architecture & Implementation Plan

**Date:** 2026-08-10
**Author:** XAheli
**Status:** Design complete, ready for implementation
**Research basis:** `drift_capture_research.md` — Strategy E backtest results

---

## 1. Mathematical Foundation

### 1.1 Derivation from First Principles

The **excess growth rate** (diversification return) for portfolio π with covariance matrix a:

$$
\gamma^*_\pi = \frac{1}{2}\left[\sum_i \pi_i a_{ii} - \pi^\top a\, \pi\right]
$$

This quantity is always non-negative for long-only portfolios (Jensen's inequality) and
represents the growth advantage from rebalancing toward a diversified portfolio vs.
buy-and-hold.

**Taking the gradient with respect to portfolio weights:**

$$
\frac{\partial \gamma^*}{\partial \pi_i}
= \frac{1}{2}\left(a_{ii} - 2\sum_j \pi_j a_{ij}\right)
= \frac{1}{2}\left(a_{ii} - 2\,a^\pi_i\right)
$$

where $a^\pi_i = (a\pi)_i$ is the covariance of stock $i$ with the portfolio.

**The gradient direction tells us:** overweight stocks whose own variance ($a_{ii}$) exceeds
twice their covariance with the current portfolio ($2a^\pi_i$). These are the stocks that
ADD to diversification — volatile but uncorrelated with the rest.

### 1.2 The Strategy

Start from market-cap weights $\mu$ and tilt in the direction of the gradient:

$$
w_i = \mu_i + \lambda \cdot \frac{\partial \gamma^*}{\partial \pi_i}\bigg|_{\pi=\mu}
= \mu_i + \frac{\lambda}{2}\left(a_{ii} - 2\,(a\mu)_i\right)
$$

Then project onto the simplex (non-negative, sum-to-one) with optional max-weight constraint.

### 1.3 Economic Interpretation

| Gradient component | Interpretation |
|---|---|
| $a_{ii}$ large | Stock has high idiosyncratic volatility → more rebalancing premium available |
| $a^\mu_i$ small | Stock has low covariance with market → diversification benefit |
| $a_{ii} - 2a^\mu_i > 0$ | Stock contributes positively to γ* when overweighted |

The strategy overweights stocks that are:
- **Volatile** (high own variance)
- **Uncorrelated** with the portfolio (low portfolio covariance)

This is mathematically identical to targeting the "diversification gap" between weighted
average variance and portfolio variance.

### 1.4 The λ Parameter

$\lambda$ (lambda_scale) controls how aggressively the portfolio deviates from market-cap:

| λ | Behavior |
|---|---|
| 0.0 | Market-cap weighted (no tilt) |
| 0.05 | Conservative — mild diversification tilt |
| 0.10 | **Research default** — +269 bps/yr alpha on 2020-2026 data |
| 0.20 | Aggressive — higher expected alpha but more tracking error |
| 0.50+ | Extreme — large deviations, may hit constraints aggressively |

**Sizing λ properly:** The gradient ∂γ*/∂π_i has units of [variance/year]. For typical
S&P 500 stocks, $a_{ii}$ ≈ 0.04-0.15 (annualized). So the gradient is O(0.01-0.05).
With λ=0.1, the weight tilt per stock is O(0.001-0.005) — a 10-50 bps shift per stock.
This keeps the portfolio close to the market while systematically harvesting the premium.

### 1.5 Constraints

The raw formula can produce negative or excessive weights. We apply:

1. **Non-negativity:** $w_i \geq 0$ (long-only)
2. **Sum-to-one:** $\sum_i w_i = 1$ (fully invested)
3. **Max position:** $w_i \leq w_{\max}$ (concentration limit, default 5%)
4. **Min position (optional):** $w_i \geq w_{\min}$ or $w_i = 0$ (cardinality)

**Simplex projection algorithm:** After computing raw weights, project onto the constrained
simplex using Duchi et al. (2008) — O(n log n) projection that satisfies all constraints
simultaneously.

---

## 2. Architectural Position in quantspt

### 2.1 Not a GeneratingFunction

The existing `GeneratingFunction` ABC produces weights via the Fernholz formula:
$\pi_i = [D_i \log G(\mu) + 1 - \sum_k \mu_k D_k \log G(\mu)] \mu_i$

This creates a **boundary term** $\log(G(\mu_T)/G(\mu_0))$ in the master formula.
The γ* gradient strategy explicitly avoids this — no G, no boundary term.

### 2.2 New Category: Direct Optimization Strategies

Create a new module `quantspt/strategies/` for strategies that target portfolio objectives
directly without a generating function. The γ* gradient strategy is the first and most
important member.

```
quantspt/
├── strategies/              ← NEW MODULE
│   ├── __init__.py
│   ├── base.py             ← Strategy protocol/ABC
│   ├── gamma_gradient.py   ← GammaGradientStrategy (THE flagship)
│   └── projections.py      ← Simplex projection utilities
├── core/
│   ├── growth_rates.py     ← Already has excess_growth_rate()
│   └── covariance.py       ← Already has tau_diagonal(), portfolio_covariance_vector()
└── backtesting/
    └── engine.py           ← Already supports WeightFunction callables
```

### 2.3 Why This Placement

- `quantspt.strategies` is a NEW top-level module — signaling this is a first-class concept
- It does NOT go under `core/generating_functions.py` — it's categorically different
- It does NOT go under `optimization/` — this is a portfolio construction strategy, not a solver
- It does NOT go under `experimental/` — this is production-ready, the RECOMMENDED approach

### 2.4 Relationship to Existing Modules

| Module | Relationship |
|---|---|
| `core.growth_rates` | Uses `excess_growth_rate()` for monitoring γ*; gradient computation inline |
| `core.covariance` | Uses `portfolio_covariance_vector()` for $a^\mu$ computation |
| `backtesting.engine` | Plugs directly via `WeightFunction` protocol |
| `backtesting.execution` | Uses `ProportionalCostExecution` for realistic costs |
| `backtesting.rebalancing` | Uses `CalendarRebalancer(MONTHLY)` by default |
| `optimization` | Does NOT use — strategy is closed-form, no numerical optimization |
| `data.providers.yfinance` | For real-data backtests |

---

## 3. API Design

### 3.1 Simple Usage (5 lines)

```python
from quantspt.strategies import GammaGradientStrategy

strategy = GammaGradientStrategy(lambda_scale=0.1, max_weight=0.05)
weights = strategy.compute_weights(market_weights, covariance_matrix)
```

### 3.2 Full Backtest

```python
from quantspt.strategies import GammaGradientStrategy

strategy = GammaGradientStrategy(
    lambda_scale=0.1,
    max_weight=0.05,
    lookback_days=126,           # 6-month rolling covariance
    rebalance_frequency="monthly",
)

result = strategy.backtest(
    prices,                       # DataFrame of daily prices (dates × stocks)
    cost_bps=10,                  # proportional transaction cost
    start_date="2020-01-01",
    end_date="2026-07-31",
)

print(f"Alpha: {result.annualized_alpha_bps:.0f} bps/yr")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Turnover: {result.annual_turnover:.1f}x")
print(result.yearly_excess)       # per-year breakdown
```

### 3.3 Integration with Existing Backtesting Engine

```python
from quantspt.backtesting import BacktestEngine, ProportionalCostExecution, CalendarRebalancer
from quantspt.strategies import GammaGradientStrategy

strategy = GammaGradientStrategy(lambda_scale=0.1, max_weight=0.05)

engine = BacktestEngine(
    weight_func=strategy.weight_function(covariance_matrix),
    returns=returns_array,
    initial_weights=initial_market_weights,
    rebalancer=CalendarRebalancer("monthly"),
    execution=ProportionalCostExecution(cost_bps=10),
)
result = engine.run()
```

### 3.4 Advanced: Time-Varying Covariance

```python
strategy = GammaGradientStrategy(
    lambda_scale=0.1,
    max_weight=0.05,
    covariance_estimator="ledoit_wolf",  # shrinkage for stability
    min_variance_floor=1e-6,             # prevent near-singular matrices
)
```

### 3.5 Class Interface

```python
class GammaGradientStrategy:
    """Direct γ* (excess growth rate) gradient targeting strategy.

    Parameters
    ----------
    lambda_scale : float, default 0.1
        Gradient step size. Controls aggressiveness of deviation from
        market-cap weights. Research default: 0.1 (+269 bps/yr).
    max_weight : float, default 0.05
        Maximum weight per stock (concentration limit).
    min_weight : float, default 0.0
        Minimum non-zero weight. Stocks below this are zeroed out.
    lookback_days : int, default 126
        Rolling window for covariance estimation (trading days).
    rebalance_frequency : str, default "monthly"
        One of "daily", "weekly", "monthly", "quarterly".
    covariance_estimator : str, default "sample"
        One of "sample", "ledoit_wolf", "exponential".
    """

    def compute_weights(self, mu, cov) -> ndarray:
        """Compute optimal weights given market weights and covariance."""

    def compute_gradient(self, mu, cov) -> ndarray:
        """Compute raw γ* gradient at the market portfolio."""

    def weight_function(self, cov) -> WeightFunction:
        """Return a WeightFunction compatible with BacktestEngine."""

    def backtest(self, prices, **kwargs) -> GammaBacktestResult:
        """Run full backtest on price data with rolling covariance."""

    def gamma_star(self, weights, cov) -> float:
        """Compute γ* for given weights (for monitoring)."""
```

---

## 4. Implementation Plan

### 4.1 Module Creation

| File | Purpose | Priority |
|---|---|---|
| `quantspt/strategies/__init__.py` | Package init, exports | P0 |
| `quantspt/strategies/base.py` | `Strategy` protocol | P0 |
| `quantspt/strategies/gamma_gradient.py` | Main implementation | P0 |
| `quantspt/strategies/projections.py` | Simplex projection | P0 |

### 4.2 Implementation Steps

1. **Simplex projection** (`projections.py`)
   - Implement Duchi et al. (2008) projection onto bounded simplex
   - Handles: non-negativity, sum-to-one, max_weight, min_weight
   - O(n log n) complexity, numerically stable

2. **Strategy base** (`base.py`)
   - Protocol defining the `Strategy` interface
   - `compute_weights(mu, cov) -> ndarray`
   - `weight_function(cov) -> WeightFunction` for BacktestEngine compatibility

3. **GammaGradientStrategy** (`gamma_gradient.py`)
   - Core gradient computation: $\partial\gamma^*/\partial\pi = \frac{1}{2}(\text{diag}(a) - 2\,a\mu)$
   - Weight construction: $w = \mu + \lambda \cdot \nabla\gamma^*$
   - Simplex projection with constraints
   - Rolling covariance backtest wrapper
   - Comprehensive input validation (NaN, negative weights, ill-conditioned cov)

4. **Top-level integration** (`quantspt/__init__.py`)
   - Export `GammaGradientStrategy` at package level
   - Add to `__all__`

### 4.3 Covariance Estimation (within the strategy)

The strategy needs to estimate the covariance matrix from historical returns. Options:

| Estimator | Use when | Properties |
|---|---|---|
| `sample` (default) | Large n/T ratio, many observations | Unbiased but noisy for n~100 |
| `ledoit_wolf` | n comparable to T | Shrinks toward identity, better conditioned |
| `exponential` | Regime changes | Recent data weighted more, adapts faster |

For the research reproduction (99 stocks, 126-day window), sample covariance suffices
because T/n ≈ 1.3. For production with 500 stocks, Ledoit-Wolf is strongly recommended.

### 4.4 Test Strategy

**Mathematical correctness tests:**
- Gradient computed correctly (finite differences vs analytical)
- γ* is non-negative for long-only portfolios
- Gradient is zero at the global max (equal-weight for diagonal cov)
- Simplex projection preserves sum-to-one, non-negativity, max-weight
- Edge cases: single stock, two stocks, identical stocks

**Backtest validation tests:**
- Reproduce +269 bps/yr on same data as research (99 S&P 500 stocks, 2020-2026)
- Cost sensitivity: result degrades gracefully with higher costs
- Lambda sensitivity: alpha monotonically increases (up to a point) with lambda
- Year-by-year breakdown matches research table (±50 bps tolerance)

**Integration tests:**
- Plugs into BacktestEngine via WeightFunction
- Works with ProportionalCostExecution
- Works with CalendarRebalancer
- Compatible with existing performance metrics

**Robustness tests:**
- Handles NaN in prices (raises clear error)
- Handles all-zero covariance row (stock with no variance)
- Handles ill-conditioned covariance (adds diagonal ridge)
- Handles single rebalance period
- Handles universe changes (stocks entering/leaving)

---

## 5. Risk Management

### 5.1 Failure Modes

| Failure mode | Cause | Mitigation |
|---|---|---|
| **Covariance estimation error** | Rolling window too short, regime change | Use shrinkage estimator, minimum lookback of 63 days |
| **Extreme concentration** | One stock has very high variance | max_weight constraint (default 5%) |
| **Negative gradient** for most stocks | Market is already very diversified (rare) | Portfolio stays near market-cap (small tilts) |
| **Tracking error blow-up** | λ too large for universe size | λ should scale inversely with √n for large universes |
| **Turnover explosion** | Covariance estimates unstable | Turnover penalty (optional), wider rebalancing window |

### 5.2 Sizing λ Properly

Rules of thumb from the research:
- **Conservative:** λ = 0.05 → ~150 bps alpha, ~2% tracking error
- **Default:** λ = 0.1 → ~269 bps alpha, ~4% tracking error
- **Aggressive:** λ = 0.2 → ~400 bps alpha, ~7% tracking error (untested)

**Adaptive λ (future enhancement):** Scale λ by the average gradient magnitude to maintain
constant tracking error regardless of market regime. Not in v1.

### 5.3 Extreme Market Events

| Event | Impact | Strategy behavior |
|---|---|---|
| **Flash crash** | Covariance spikes, gradients become large | max_weight constraint binds; portfolio stays bounded |
| **Concentration surge** (NVDA 2024) | Market weights shift rapidly | Strategy adapts next rebalance; no boundary term to damage |
| **Correlation spike** (March 2020) | All covariances increase | Gradient magnitude decreases (less diversification available); strategy tilts less aggressively |
| **Circuit breakers** | Missing data | Use last valid covariance estimate |

### 5.4 Combining with Universe Selection

The strategy works best with:
- **Liquid stocks only** (S&P 500, Russell 1000) — ensures tradability
- **Exclude recent IPOs** — insufficient history for covariance estimation
- **Sector-neutral (optional)** — compute gradient within sectors to avoid sector bets

Integration with `quantspt.universe.selector`:
```python
from quantspt.universe import UniverseSelector, LiquidityCriteria

universe = UniverseSelector(criteria=[LiquidityCriteria(min_adv=1e6)])
strategy = GammaGradientStrategy(lambda_scale=0.1)
# strategy.backtest() can accept a universe selector
```

---

## 6. Validation Plan

### 6.1 In-Sample Reproduction (MUST PASS)

**Data:** 99 S&P 500 stocks, Jan 2020 – Jul 2026 (same as research)
**Parameters:** λ=0.1, max_weight=0.05, 126-day lookback, monthly rebalance, 10bps cost
**Expected:** +269 bps/yr annualized alpha (β-adjusted), ±30 bps tolerance
**Year breakdown must approximate:**

| Year | Expected excess (bps) | Tolerance |
|---|---|---|
| 2020 | +454 | ±100 |
| 2021 | +91 | ±100 |
| 2022 | -169 | ±100 |
| 2023 | +368 | ±100 |
| 2024 | +3000 | ±200 |
| 2025 | +479 | ±100 |

### 6.2 Walk-Forward Out-of-Sample

**Split:** Train covariance model on 2020-2023, test on 2024-2026
**Purpose:** Verify no look-ahead bias in covariance estimation
**Expected:** Positive alpha in test period (2024-2026 should be strong based on research)

### 6.3 Multiple Universes

| Universe | Expected behavior |
|---|---|
| S&P 500 (full, ~500 stocks) | Should work — more diversification opportunities |
| Russell 1000 | Should work — broader universe |
| Top 50 only | Lower alpha (less diversification room) |
| International (if data available) | Unknown — validation needed |

### 6.4 Cost Sensitivity

| Cost (bps) | Expected impact |
|---|---|
| 0 | Baseline (highest alpha) |
| 5 | ~5 bps alpha reduction |
| 10 | Research default — ~269 bps |
| 20 | ~260 bps (low sensitivity due to low turnover) |
| 50 | ~240 bps |

**Why low cost sensitivity:** Strategy has ~2x annual turnover on liquid S&P 500 names.
At 10bps cost, total cost is ~20bps/yr — small relative to 269bps alpha.

### 6.5 Comparison Benchmarks

| Benchmark | Expected relative performance |
|---|---|
| Market-cap weighted | +269 bps/yr alpha (by definition) |
| Equal weight | +188 bps advantage (EW has size factor bias) |
| DiversityGenerator(p=0.3) | +210 bps better (Diversity lost -246bps on this data) |
| Minimum variance | Similar risk, but γ* gradient has theoretical grounding |

---

## 7. Future Enhancements (NOT in v1)

1. **Adaptive λ:** Target constant tracking error via λ = target_TE / gradient_magnitude
2. **Factor-neutral:** Orthogonalize gradient to known factors (market, size, value)
3. **Multi-period optimization:** Account for turnover in weight computation
4. **Regime detection:** Scale λ based on detected volatility regime
5. **130/30 extension:** Allow limited shorting for stocks with very negative gradient
6. **Sector constraints:** Equal-sector deviation from market within each GICS sector
7. **Tax-aware rebalancing:** Loss harvesting integration for taxable accounts

---

## 8. Implementation Timeline

| Step | Deliverable | Commit message |
|---|---|---|
| 1 | This plan | `docs: add gamma gradient strategy design plan` |
| 2 | Core implementation | `feat(strategy): implement GammaGradientStrategy with direct γ* targeting` |
| 3 | In-sample validation | `test(strategy): validate +269 bps alpha reproduction on S&P 500 2020-2026` |
| 4 | Walk-forward test | `test(strategy): walk-forward out-of-sample validation` |
| 5 | Top-level integration | `feat: integrate gamma gradient as recommended strategy in top-level API` |

---

## Appendix A: Simplex Projection Algorithm

Given raw weights $w$ and constraints $0 \leq w_i \leq w_{\max}$, $\sum w_i = 1$:

1. Sort $w$ in descending order
2. Find threshold $\theta$ such that $\sum_i \min(\max(w_i - \theta, 0), w_{\max}) = 1$
3. Apply: $w_i^* = \min(\max(w_i - \theta, 0), w_{\max})$

This is the Euclidean projection (minimum ||w - w*||²) — numerically stable, O(n log n).

## Appendix B: Connection to Existing SPT Concepts

The gradient $\partial\gamma^*/\partial\pi_i = \frac{1}{2}(a_{ii} - 2a^\pi_i)$ relates to:

- **τ diagonal:** $\tau^\pi_{ii} = a_{ii} - 2a^\pi_i + a_{\pi\pi}$, so $\partial\gamma^*/\partial\pi_i = \frac{1}{2}(\tau^\pi_{ii} - a_{\pi\pi})$
- **Excess growth rate:** $\gamma^* = \frac{1}{2}\sum_i \pi_i \tau^\pi_{ii}$ (the numeraire-invariant form)
- **The strategy tilts toward stocks with high τ^μ_{ii}** — those with high relative variance vs. the market

This connects the strategy back to the core SPT apparatus in `quantspt.core`.
