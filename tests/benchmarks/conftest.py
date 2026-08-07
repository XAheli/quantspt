"""Shared fixtures for benchmark tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats


def _pareto_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    raw = (rng.pareto(1.0, size=n) + 1.0).astype(np.float64)
    return raw / raw.sum()


def _sector_covariance(
    rng: np.random.Generator,
    n: int,
    n_sectors: int = 5,
    intra_corr: float = 0.6,
    inter_corr: float = 0.3,
    vol_range: tuple[float, float] = (0.15, 0.40),
) -> np.ndarray:
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


def _realistic_returns(
    rng: np.random.Generator,
    n_assets: int,
    n_days: int,
) -> np.ndarray:
    dt = 1.0 / 252.0
    daily_mean = 0.08 * dt
    vols = rng.uniform(0.15, 0.40, size=n_assets)
    daily_std = vols * np.sqrt(dt)
    raw = stats.t.rvs(df=5.0, size=(n_days, n_assets), random_state=rng)
    raw = raw - raw.mean(axis=0)
    raw = raw / raw.std(axis=0)
    returns = daily_mean + daily_std * raw
    return returns.astype(np.float64)


SCENARIOS = {
    "small": {"n_stocks": 10, "n_days": 252, "n_paths": 1_000},
    "medium": {"n_stocks": 50, "n_days": 1_260, "n_paths": 10_000},
    "large": {"n_stocks": 500, "n_days": 2_520, "n_paths": 10_000},
    "stress": {"n_stocks": 500, "n_days": 2_520, "n_paths": 100_000},
    "massive": {"n_stocks": 2_000, "n_days": 5_040, "n_paths": 10_000},
}


@pytest.fixture(scope="module")
def scenario_data():
    """Pre-generate data for all scenarios."""
    data = {}
    for name, params in SCENARIOS.items():
        n = params["n_stocks"]
        n_days = params["n_days"]
        rng = np.random.default_rng(42)
        data[name] = {
            "pi": _pareto_weights(rng, n),
            "cov": _sector_covariance(rng, n, n_sectors=min(n, 11)),
            "returns": _realistic_returns(rng, n, n_days),
            "n_stocks": n,
            "n_days": n_days,
            "n_paths": params["n_paths"],
        }
    return data
