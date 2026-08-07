"""Tests for data/cache.py — lazy evaluation with dirty-flag invalidation."""

from __future__ import annotations

import threading
import time

import pytest

from quantspt.data.cache import CachedComputation, ComputationCache
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# CachedComputation tests
# ---------------------------------------------------------------------------


class TestCachedComputation:
    def test_basic_computation(self) -> None:
        cache = CachedComputation(lambda: 42, name="answer")
        assert cache.get() == 42

    def test_caches_result(self) -> None:
        call_count = [0]

        def expensive():
            call_count[0] += 1
            return sum(range(100))

        cache = CachedComputation(expensive, name="sum")
        assert cache.get() == 4950
        assert cache.get() == 4950
        assert call_count[0] == 1

    def test_starts_dirty(self) -> None:
        cache = CachedComputation(lambda: 1, name="x")
        assert cache.is_dirty

    def test_clean_after_get(self) -> None:
        cache = CachedComputation(lambda: 1, name="x")
        cache.get()
        assert not cache.is_dirty

    def test_invalidate_marks_dirty(self) -> None:
        cache = CachedComputation(lambda: 1, name="x")
        cache.get()
        assert not cache.is_dirty
        cache.invalidate()
        assert cache.is_dirty

    def test_recomputes_after_invalidation(self) -> None:
        counter = [0]

        def compute():
            counter[0] += 1
            return counter[0]

        cache = CachedComputation(compute, name="counter")
        assert cache.get() == 1
        assert cache.get() == 1
        cache.invalidate()
        assert cache.get() == 2

    def test_version_increments(self) -> None:
        cache = CachedComputation(lambda: 1, name="x")
        assert cache.version == 0
        cache.get()
        assert cache.version == 1
        cache.invalidate()
        cache.get()
        assert cache.version == 2

    def test_dependency_tracking(self) -> None:
        upstream_val = [10]
        upstream = CachedComputation(lambda: upstream_val[0], name="upstream")
        downstream = CachedComputation(
            lambda: upstream.get() * 2,
            name="downstream",
            dependencies=[upstream],
        )

        assert downstream.get() == 20
        assert not downstream.is_dirty

        upstream_val[0] = 20
        upstream.invalidate()
        upstream.get()  # recomputes upstream, bumps version

        assert downstream.is_dirty
        assert downstream.get() == 40

    def test_peek_returns_none_when_dirty(self) -> None:
        cache = CachedComputation(lambda: 42, name="x")
        assert cache.peek() is None
        cache.get()
        assert cache.peek() == 42
        cache.invalidate()
        assert cache.peek() is None

    def test_set_compute_fn(self) -> None:
        cache = CachedComputation(lambda: 1, name="x")
        assert cache.get() == 1
        cache.set_compute_fn(lambda: 99)
        assert cache.is_dirty
        assert cache.get() == 99

    def test_stats_tracking(self) -> None:
        cache = CachedComputation(lambda: 42, name="x")
        cache.get()  # miss
        cache.get()  # hit
        cache.get()  # hit
        cache.invalidate()
        cache.get()  # miss

        assert cache.stats.hits == 2
        assert cache.stats.misses == 2
        assert cache.stats.invalidations == 1
        assert cache.stats.hit_rate == 0.5

    def test_last_compute_time(self) -> None:
        def slow():
            time.sleep(0.01)
            return 1

        cache = CachedComputation(slow, name="slow")
        cache.get()
        assert cache.last_compute_time >= 0.005

    def test_add_dependency(self) -> None:
        a = CachedComputation(lambda: 1, name="a")
        b = CachedComputation(lambda: a.get() + 1, name="b")

        b.get()
        assert b.get() == 2

        b.add_dependency(a)
        a.invalidate()
        a.get()
        assert b.is_dirty

    def test_remove_dependency(self) -> None:
        a = CachedComputation(lambda: 1, name="a")
        b = CachedComputation(lambda: 2, name="b", dependencies=[a])
        b.get()
        b.remove_dependency(a)
        a.invalidate()
        a.get()
        assert not b.is_dirty

    def test_non_callable_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="callable"):
            CachedComputation(42, name="bad")  # type: ignore[arg-type]

    def test_thread_safety(self) -> None:
        counter = [0]
        lock = threading.Lock()

        def compute():
            with lock:
                counter[0] += 1
            time.sleep(0.001)
            return counter[0]

        cache = CachedComputation(compute, name="threaded")
        results: list[int] = []

        def reader():
            results.append(cache.get())

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == results[0] for r in results)
        assert counter[0] == 1

    def test_concurrent_invalidate_and_get(self) -> None:
        counter = [0]

        def compute():
            counter[0] += 1
            return counter[0]

        cache = CachedComputation(compute, name="concurrent")
        cache.get()

        errors: list[Exception] = []

        def invalidate_loop():
            try:
                for _ in range(50):
                    cache.invalidate()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def get_loop():
            try:
                for _ in range(50):
                    cache.get()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=invalidate_loop)
        t2 = threading.Thread(target=get_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0

    def test_name_property(self) -> None:
        cache = CachedComputation(lambda: 1, name="my_cache")
        assert cache.name == "my_cache"

    def test_stats_hit_rate_zero_when_empty(self) -> None:
        cache = CachedComputation(lambda: 1, name="x")
        assert cache.stats.hit_rate == 0.0

    def test_concurrent_is_dirty_and_invalidate(self) -> None:
        """is_dirty must not race with invalidate/get from other threads."""
        counter = [0]

        def compute():
            counter[0] += 1
            return counter[0]

        cache = CachedComputation(compute, name="race")
        cache.get()

        errors: list[Exception] = []
        dirty_results: list[bool] = []

        def check_dirty_loop():
            try:
                for _ in range(200):
                    dirty_results.append(cache.is_dirty)
            except Exception as e:
                errors.append(e)

        def invalidate_and_get_loop():
            try:
                for _ in range(200):
                    cache.invalidate()
                    cache.get()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=check_dirty_loop)
        t2 = threading.Thread(target=invalidate_and_get_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
        assert all(isinstance(d, bool) for d in dirty_results)

    def test_is_dirty_uses_public_version(self) -> None:
        """is_dirty should use dep.version (public), not dep._version."""
        upstream = CachedComputation(lambda: 1, name="up")
        downstream = CachedComputation(
            lambda: upstream.get() + 1,
            name="down",
            dependencies=[upstream],
        )
        downstream.get()
        assert not downstream.is_dirty
        assert downstream.version == 1
        assert upstream.version == 1


# ---------------------------------------------------------------------------
# ComputationCache tests
# ---------------------------------------------------------------------------


class TestComputationCache:
    def test_register_and_get(self) -> None:
        cache = ComputationCache()
        cache.register("x", lambda: 42)
        assert cache.get("x") == 42

    def test_caches_values(self) -> None:
        counter = [0]

        def compute():
            counter[0] += 1
            return counter[0]

        cache = ComputationCache()
        cache.register("c", compute)
        assert cache.get("c") == 1
        assert cache.get("c") == 1
        assert counter[0] == 1

    def test_invalidate_specific(self) -> None:
        counter = [0]

        def compute():
            counter[0] += 1
            return counter[0]

        cache = ComputationCache()
        cache.register("c", compute)
        cache.get("c")
        cache.invalidate("c")
        assert cache.get("c") == 2

    def test_invalidate_all(self) -> None:
        cache = ComputationCache()
        counters = {"a": [0], "b": [0]}

        cache.register(
            "a",
            lambda: (
                counters["a"].__setitem__(0, counters["a"][0] + 1) or counters["a"][0]
            ),
        )
        cache.register(
            "b",
            lambda: (
                counters["b"].__setitem__(0, counters["b"][0] + 1) or counters["b"][0]
            ),
        )

        cache.get("a")
        cache.get("b")
        cache.invalidate_all()
        cache.get("a")
        cache.get("b")

        assert counters["a"][0] == 2
        assert counters["b"][0] == 2

    def test_dependency_chain(self) -> None:
        cache = ComputationCache()
        base_val = [10]

        cache.register("base", lambda: base_val[0])
        cache.register("derived", lambda: cache.get("base") * 2, dependencies=["base"])

        assert cache.get("derived") == 20

        base_val[0] = 50
        cache.invalidate("base")
        assert cache.get("derived") == 100

    def test_unknown_name_raises(self) -> None:
        cache = ComputationCache()
        with pytest.raises(SPTInvariantError, match="not registered"):
            cache.get("nonexistent")

    def test_unknown_dependency_raises(self) -> None:
        cache = ComputationCache()
        with pytest.raises(SPTInvariantError, match="not registered"):
            cache.register("x", lambda: 1, dependencies=["missing"])

    def test_names_property(self) -> None:
        cache = ComputationCache()
        cache.register("a", lambda: 1)
        cache.register("b", lambda: 2)
        assert set(cache.names) == {"a", "b"}

    def test_contains(self) -> None:
        cache = ComputationCache()
        cache.register("x", lambda: 1)
        assert "x" in cache
        assert "y" not in cache

    def test_len(self) -> None:
        cache = ComputationCache()
        assert len(cache) == 0
        cache.register("a", lambda: 1)
        assert len(cache) == 1

    def test_stats_aggregation(self) -> None:
        cache = ComputationCache()
        cache.register("x", lambda: 1)
        cache.get("x")
        cache.get("x")
        stats = cache.stats()
        assert stats["x"].hits == 1
        assert stats["x"].misses == 1

    def test_invalidate_unknown_raises(self) -> None:
        cache = ComputationCache()
        with pytest.raises(SPTInvariantError, match="not registered"):
            cache.invalidate("ghost")
