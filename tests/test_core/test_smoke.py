"""Smoke tests: verify the package imports and basic wiring."""

from __future__ import annotations


def test_import_quantspt() -> None:
    import quantspt

    assert hasattr(quantspt, "__version__")


def test_version_string() -> None:
    from quantspt._version import __version__

    assert __version__ == "0.1.0"


def test_spt_result_basic() -> None:
    import numpy as np

    from quantspt._result import SPTResult

    result: SPTResult[np.ndarray] = SPTResult(data=np.array([1.0, 2.0]))
    assert result.validate()
    assert "SPTResult" in repr(result)


def test_preconditions() -> None:
    import pytest

    from quantspt._preconditions import ensure, require
    from quantspt.errors import SPTInvariantError

    require(True, "ok")
    with pytest.raises(SPTInvariantError, match="Precondition"):
        require(False, "weights must sum to 1")

    ensure(True, "ok")
    with pytest.raises(SPTInvariantError, match="Postcondition"):
        ensure(False, "result must be non-negative")


def test_error_hierarchy() -> None:
    from quantspt.errors import (
        InfeasibleError,
        OptimizationError,
        SPTError,
        SPTInvariantError,
    )

    assert issubclass(SPTInvariantError, SPTError)
    assert issubclass(InfeasibleError, OptimizationError)
    assert issubclass(OptimizationError, SPTError)


def test_config() -> None:
    from quantspt._config import get_config

    cfg = get_config()
    assert cfg.backend == "numpy"


def test_close_utility() -> None:
    from quantspt.utils import close

    assert close(1.0, 1.0 + 1e-15)
    assert not close(1.0, 2.0)
