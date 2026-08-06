"""Shared fixtures for quantspt tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture()
def sample_weights_2() -> np.ndarray:
    """Simple 2-asset weight vector."""
    return np.array([0.6, 0.4])


@pytest.fixture()
def sample_weights_5(rng: np.random.Generator) -> np.ndarray:
    """Dirichlet-sampled 5-asset weight vector."""
    return rng.dirichlet(np.ones(5))


@pytest.fixture()
def diagonal_cov_2() -> np.ndarray:
    """2×2 diagonal covariance matrix."""
    return np.diag([0.04, 0.09])


@pytest.fixture()
def psd_cov_5(rng: np.random.Generator) -> np.ndarray:
    """Random 5×5 PSD covariance matrix."""
    L = rng.standard_normal((5, 5))
    return L @ L.T + np.eye(5) * 0.01


# ---------------------------------------------------------------------------
# Realistic market weight fixtures (Pareto-distributed, various sizes)
# ---------------------------------------------------------------------------


def _pareto_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate Pareto-distributed market weights (power-law, top-heavy)."""
    raw = (rng.pareto(1.0, size=n) + 1.0).astype(np.float64)
    return raw / raw.sum()


@pytest.fixture()
def pareto_weights_5() -> np.ndarray:
    """Pareto-distributed 5-asset market weights (top stock ~30-50%)."""
    return _pareto_weights(np.random.default_rng(42), 5)


@pytest.fixture()
def pareto_weights_10() -> np.ndarray:
    """Pareto-distributed 10-asset market weights."""
    return _pareto_weights(np.random.default_rng(42), 10)


@pytest.fixture()
def pareto_weights_50() -> np.ndarray:
    """Pareto-distributed 50-asset market weights (top stock ~5-15%)."""
    return _pareto_weights(np.random.default_rng(42), 50)


@pytest.fixture()
def pareto_weights_100() -> np.ndarray:
    """Pareto-distributed 100-asset market weights (large-cap universe)."""
    return _pareto_weights(np.random.default_rng(42), 100)


@pytest.fixture()
def dominant_stock_weights() -> np.ndarray:
    """Edge case: one dominant stock at ~90%."""
    w = np.array([0.90, 0.04, 0.03, 0.02, 0.01])
    return w / w.sum()


@pytest.fixture()
def near_equal_weights_5() -> np.ndarray:
    """Edge case: near-equal weights (close to 1/n)."""
    rng = np.random.default_rng(42)
    w = np.ones(5) / 5.0 + rng.normal(0, 1e-4, 5)
    w = np.abs(w)
    return w / w.sum()


@pytest.fixture()
def near_zero_weights() -> np.ndarray:
    """Edge case: some weights very close to zero."""
    w = np.array([0.50, 0.30, 0.15, 0.04, 1e-6])
    return w / w.sum()


# ---------------------------------------------------------------------------
# Realistic covariance matrices
# ---------------------------------------------------------------------------


def _sector_covariance(
    rng: np.random.Generator,
    n: int,
    n_sectors: int = 3,
    intra_corr: float = 0.6,
    inter_corr: float = 0.3,
    vol_range: tuple[float, float] = (0.15, 0.40),
) -> np.ndarray:
    """Generate a block-diagonal sector-structured covariance matrix.

    Mimics real equity markets: stocks within the same sector have higher
    correlation (~0.5-0.7) than stocks across sectors (~0.2-0.4).
    """
    sector_sizes = np.diff(np.linspace(0, n, n_sectors + 1, dtype=int))
    vols = rng.uniform(vol_range[0], vol_range[1], size=n)

    corr = np.full((n, n), inter_corr)
    offset = 0
    for sz in sector_sizes:
        corr[offset : offset + sz, offset : offset + sz] = intra_corr
        offset += sz
    np.fill_diagonal(corr, 1.0)

    D = np.diag(vols)
    cov = D @ corr @ D
    cov = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals[0] < 0:
        cov += np.eye(n) * (-eigvals[0] + 1e-8)
    return cov


@pytest.fixture()
def sector_cov_5() -> np.ndarray:
    """5×5 sector-structured covariance (2 sectors, realistic correlations)."""
    return _sector_covariance(np.random.default_rng(42), 5, n_sectors=2)


@pytest.fixture()
def sector_cov_10() -> np.ndarray:
    """10×10 sector-structured covariance (3 sectors)."""
    return _sector_covariance(np.random.default_rng(42), 10, n_sectors=3)


@pytest.fixture()
def sector_cov_50() -> np.ndarray:
    """50×50 sector-structured covariance (5 sectors)."""
    return _sector_covariance(np.random.default_rng(42), 50, n_sectors=5)


# ---------------------------------------------------------------------------
# Realistic return data
# ---------------------------------------------------------------------------


def _realistic_returns(
    rng: np.random.Generator,
    n_assets: int,
    n_days: int,
    annual_mean: float = 0.08,
    annual_vol_range: tuple[float, float] = (0.15, 0.40),
    skew: float = -0.3,
    excess_kurtosis: float = 2.0,
) -> np.ndarray:
    """Generate realistic daily returns with fat tails and slight negative skew.

    Uses a skew-t distribution to match empirical market microstructure:
    - Daily mean ~0.0003 (annualized ~8%)
    - Daily std ~0.01-0.025 (annualized 15-40%)
    - Slight negative skew (crash risk)
    - Fat tails (excess kurtosis ~2-4)
    """
    dt = 1.0 / 252.0
    daily_mean = annual_mean * dt
    vols = rng.uniform(annual_vol_range[0], annual_vol_range[1], size=n_assets)
    daily_std = vols * np.sqrt(dt)

    df_t = 2.0 / excess_kurtosis + 4.0 if excess_kurtosis > 0 else 30.0
    df_t = max(df_t, 4.1)

    raw = stats.t.rvs(df=df_t, size=(n_days, n_assets), random_state=rng)
    raw = raw - raw.mean(axis=0)
    raw = raw / raw.std(axis=0)

    if skew != 0:
        raw = raw + skew * (raw**2 - 1) / 6.0

    returns = daily_mean + daily_std * raw
    return returns.astype(np.float64)


@pytest.fixture()
def realistic_returns_1y() -> np.ndarray:
    """252 days of realistic daily returns for 10 stocks.

    Properties: ~0.0003 daily mean, ~0.015 daily std,
    slight negative skew, fat tails.
    """
    return _realistic_returns(np.random.default_rng(42), n_assets=10, n_days=252)


@pytest.fixture()
def realistic_returns_5y() -> np.ndarray:
    """1260 days (5 years) of realistic daily returns for 10 stocks."""
    return _realistic_returns(np.random.default_rng(42), n_assets=10, n_days=1260)


@pytest.fixture()
def realistic_returns_5_stocks_1y() -> np.ndarray:
    """252 days of realistic daily returns for 5 stocks."""
    return _realistic_returns(np.random.default_rng(42), n_assets=5, n_days=252)


# ---------------------------------------------------------------------------
# Realistic market fixtures (combined weights + covariance + returns)
# ---------------------------------------------------------------------------


@pytest.fixture()
def realistic_market_5() -> dict:
    """5-stock market with Pareto weights, sector covariance, and 1y returns."""
    rng = np.random.default_rng(42)
    return {
        "weights": _pareto_weights(rng, 5),
        "covariance": _sector_covariance(rng, 5, n_sectors=2),
        "returns": _realistic_returns(rng, n_assets=5, n_days=252),
        "n_assets": 5,
    }


@pytest.fixture()
def realistic_market_50() -> dict:
    """50-stock market with sector structure, Pareto weights, and 1y returns."""
    rng = np.random.default_rng(42)
    return {
        "weights": _pareto_weights(rng, 50),
        "covariance": _sector_covariance(rng, 50, n_sectors=5),
        "returns": _realistic_returns(rng, n_assets=50, n_days=252),
        "n_assets": 50,
    }
