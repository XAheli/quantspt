"""Tests for _config.py -- global configuration and backend selection."""

from __future__ import annotations

import threading

import pytest

from quantspt._config import _GLOBAL_CONFIG, get_config, set_backend


class TestGetConfig:
    def test_returns_singleton(self) -> None:
        cfg = get_config()
        assert cfg is _GLOBAL_CONFIG

    def test_default_backend(self) -> None:
        cfg = get_config()
        assert cfg.backend == "numpy" or cfg.backend in {
            "numpy",
            "numba",
            "jax",
            "cupy",
        }

    def test_default_tolerance(self) -> None:
        cfg = get_config()
        assert cfg.float_tolerance_ulps == 42


class TestSetBackend:
    def setup_method(self) -> None:
        self._original = _GLOBAL_CONFIG.backend

    def teardown_method(self) -> None:
        _GLOBAL_CONFIG.backend = self._original

    def test_set_numpy(self) -> None:
        set_backend("numpy")
        assert get_config().backend == "numpy"

    def test_set_numba(self) -> None:
        set_backend("numba")
        assert get_config().backend == "numba"

    def test_set_jax(self) -> None:
        set_backend("jax")
        assert get_config().backend == "jax"

    def test_set_cupy(self) -> None:
        set_backend("cupy")
        assert get_config().backend == "cupy"

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("tensorflow")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("")

    def test_concurrent_set_backend(self) -> None:
        """Concurrent set_backend calls must not corrupt state."""
        errors: list[Exception] = []
        backends = ["numpy", "numba", "jax", "cupy"]

        def switch_loop(backend: str) -> None:
            try:
                for _ in range(100):
                    set_backend(backend)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=switch_loop, args=(b,)) for b in backends]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert get_config().backend in {"numpy", "numba", "jax", "cupy"}
