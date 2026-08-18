"""Lazy evaluation cache with dirty-flag invalidation.

Provides a thread-safe caching mechanism for expensive computations
(e.g., covariance matrices, rank decompositions) that automatically
invalidates when upstream inputs change.

The ``CachedComputation`` class tracks dependencies and uses a dirty flag
to avoid redundant recomputation while ensuring correctness when inputs
are modified.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .._preconditions import require

__all__ = [
    "CachedComputation",
    "ComputationCache",
]

T = TypeVar("T")


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    total_compute_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Fraction of accesses served from cache."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class CachedComputation(Generic[T]):
    """A lazily-evaluated computation with dirty-flag invalidation.

    Wraps an expensive callable and caches its result. The cache is
    invalidated (marked dirty) when any tracked input changes, and
    the computation is re-executed only on the next access.

    Thread-safe: concurrent reads return the cached value while a single
    writer recomputes if dirty.

    Parameters
    ----------
    compute_fn : callable
        Function that produces the cached value. Called with no arguments.
        Captured inputs should be accessed via closure or instance reference.
    name : str
        Human-readable name for debugging and logging.
    dependencies : list of CachedComputation, optional
        Upstream computations. If any dependency is dirty or has been
        recomputed since this node's last computation, this node is
        also considered dirty.

    Examples
    --------
    >>> cache = CachedComputation(lambda: sum(range(1000)), name="sum")
    >>> cache.get()  # computes
    499500
    >>> cache.get()  # returns cached
    499500
    >>> cache.invalidate()
    >>> cache.is_dirty
    True
    """

    def __init__(
        self,
        compute_fn: Any,
        name: str = "unnamed",
        dependencies: list[CachedComputation[Any]] | None = None,
    ) -> None:
        require(callable(compute_fn), "compute_fn must be callable")
        self._compute_fn = compute_fn
        self._name = name
        self._dependencies: list[CachedComputation[Any]] = dependencies or []
        self._value: T | None = None
        self._dirty: bool = True
        self._version: int = 0
        self._dep_versions: dict[int, int] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._last_compute_time: float = 0.0

    @property
    def name(self) -> str:
        """Human-readable name."""
        return self._name

    @property
    def is_dirty(self) -> bool:
        """Whether the cache needs recomputation."""
        with self._lock:
            if self._dirty:
                return True
            for dep in self._dependencies:
                if dep.is_dirty:
                    return True
                if dep.version != self._dep_versions.get(id(dep), -1):
                    return True
            return False

    @property
    def version(self) -> int:
        """Monotonically increasing version counter."""
        return self._version

    @property
    def stats(self) -> CacheStats:
        """Cache performance statistics."""
        return self._stats

    @property
    def last_compute_time(self) -> float:
        """Wall-clock time of the most recent computation in seconds."""
        return self._last_compute_time

    def get(self) -> T:
        """Retrieve the cached value, recomputing if dirty.

        Returns
        -------
        T
            The computed value.
        """
        with self._lock:
            if self.is_dirty:
                self._recompute()
                self._stats.misses += 1
            else:
                self._stats.hits += 1
            assert self._value is not None
            return self._value

    def invalidate(self) -> None:
        """Mark this computation as needing recomputation.

        Does NOT propagate to dependents — they detect staleness via
        version checks on their next access.
        """
        with self._lock:
            self._dirty = True
            self._stats.invalidations += 1

    def set_compute_fn(self, compute_fn: Any) -> None:
        """Replace the compute function and invalidate.

        Parameters
        ----------
        compute_fn : callable
            New function to produce the cached value.
        """
        require(callable(compute_fn), "compute_fn must be callable")
        with self._lock:
            self._compute_fn = compute_fn
            self._dirty = True

    def add_dependency(self, dep: CachedComputation[Any]) -> None:
        """Add an upstream dependency.

        Parameters
        ----------
        dep : CachedComputation
            Upstream computation to track.
        """
        with self._lock:
            if dep not in self._dependencies:
                self._dependencies.append(dep)
                self._dirty = True

    def remove_dependency(self, dep: CachedComputation[Any]) -> None:
        """Remove an upstream dependency.

        Parameters
        ----------
        dep : CachedComputation
            Upstream computation to stop tracking.
        """
        with self._lock:
            if dep in self._dependencies:
                self._dependencies.remove(dep)

    def peek(self) -> T | None:
        """Return the cached value without recomputing.

        Returns
        -------
        T or None
            Cached value, or ``None`` if never computed or dirty.
        """
        with self._lock:
            if self.is_dirty:
                return None
            return self._value

    def _recompute(self) -> None:
        """Execute the compute function and update state."""
        start = time.perf_counter()
        self._value = self._compute_fn()
        elapsed = time.perf_counter() - start
        self._last_compute_time = elapsed
        self._stats.total_compute_time += elapsed
        self._dirty = False
        self._version += 1
        for dep in self._dependencies:
            self._dep_versions[id(dep)] = dep._version


@dataclass
class ComputationCache:
    """Registry of named cached computations.

    Provides a centralized store for managing multiple interdependent
    cached computations, with bulk invalidation and statistics.

    Examples
    --------
    >>> cache = ComputationCache()
    >>> cache.register("cov", lambda: expensive_cov())
    >>> cache.get("cov")  # computes on first access
    >>> cache.invalidate_all()  # mark everything dirty
    """

    _computations: dict[str, CachedComputation[Any]] = field(
        default_factory=dict, init=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def register(
        self,
        name: str,
        compute_fn: Any,
        dependencies: list[str] | None = None,
    ) -> CachedComputation[Any]:
        """Register a named computation.

        Parameters
        ----------
        name : str
            Unique name for this computation.
        compute_fn : callable
            Function to produce the value.
        dependencies : list of str, optional
            Names of upstream computations in this cache.

        Returns
        -------
        CachedComputation
            The registered computation node.
        """
        with self._lock:
            deps = []
            if dependencies:
                for dep_name in dependencies:
                    require(
                        dep_name in self._computations,
                        f"Dependency '{dep_name}' not registered",
                    )
                    deps.append(self._computations[dep_name])

            node: CachedComputation[Any] = CachedComputation(
                compute_fn=compute_fn,
                name=name,
                dependencies=deps,
            )
            self._computations[name] = node
            return node

    def get(self, name: str) -> Any:
        """Retrieve the value of a named computation.

        Parameters
        ----------
        name : str
            Name of the registered computation.

        Returns
        -------
        Any
            The computed (or cached) value.
        """
        require(
            name in self._computations,
            f"Computation '{name}' not registered",
        )
        return self._computations[name].get()

    def invalidate(self, name: str) -> None:
        """Invalidate a specific computation.

        Parameters
        ----------
        name : str
            Name of the computation to invalidate.
        """
        require(
            name in self._computations,
            f"Computation '{name}' not registered",
        )
        self._computations[name].invalidate()

    def invalidate_all(self) -> None:
        """Invalidate all registered computations."""
        with self._lock:
            for comp in self._computations.values():
                comp.invalidate()

    @property
    def names(self) -> list[str]:
        """List of registered computation names."""
        return list(self._computations.keys())

    def stats(self) -> dict[str, CacheStats]:
        """Aggregate statistics for all computations.

        Returns
        -------
        dict mapping name to CacheStats.
        """
        return {name: comp.stats for name, comp in self._computations.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._computations

    def __len__(self) -> int:
        return len(self._computations)
