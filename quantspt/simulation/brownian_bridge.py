r"""Brownian bridge path construction via binary-tree refinement.

Transforms N independent standard normal variates into a Brownian path
on an arbitrary time grid with the correct joint Gaussian distribution.
The construction assigns the terminal value first, then fills interior
points by dyadic bisection so that the most significant path features
(terminal value, then midpoints of progressively shorter intervals)
consume the lowest-indexed input variates.

This ordering is critical for quasi-random (low-discrepancy) Monte Carlo:
coordinates with small index carry the most weight in a Sobol or Halton
sequence, so they should control the most important path features.

The algorithm precomputes a table of indices and weights so that the
transform itself is a single O(N) pass of multiply-accumulate operations.

Mathematical References
-----------------------
- Bridge construction: Jäckel, *Monte Carlo Methods in Finance* (2002), Ch. 10
- Covariance of Brownian motion: Cov(W(s), W(t)) = min(s, t)
- Conditional midpoint: W(m) | W(l), W(r)
  ~ N(w_l · W(l) + w_r · W(r), σ²)
  where m = (l+r)/2, w_l = (r-m)/(r-l), w_r = (m-l)/(r-l), σ² = (m-l)(r-m)/(r-l)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class BrownianBridge:
    r"""Binary-tree Brownian bridge for path construction.

    Transforms N independent N(0,1) variates into a Brownian path on a
    time grid with correct joint distribution.  Terminal value is assigned
    first, then midpoints via dyadic refinement.

    This enables quasi-random (Sobol) Monte Carlo where the best
    low-discrepancy coordinates control the most important path features.

    The precomputed tables store, for each construction step i:
      - ``bridge_index[i]``: which path point is filled at step i
      - ``left_index[i]``, ``right_index[i]``: conditioning neighbors
      - ``left_weight[i]``, ``right_weight[i]``: interpolation weights
      - ``stddev[i]``: conditional standard deviation

    Parameters
    ----------
    times : ndarray of shape (N+1,)
        Time grid including t=0.  Must be strictly increasing with
        times[0] = 0.

    Reference: Jäckel, *Monte Carlo Methods in Finance* (2002), Ch. 10
    """

    def __init__(self, times: NDArray[np.float64]) -> None:
        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("times must be a 1-D array with at least 2 elements")
        if times[0] != 0.0:
            raise ValueError("times[0] must be 0")
        if not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing")

        self._times = times
        n = len(times) - 1

        bridge_index = np.zeros(n, dtype=np.intp)
        left_index = np.zeros(n, dtype=np.intp)
        right_index = np.zeros(n, dtype=np.intp)
        left_weight = np.zeros(n, dtype=np.float64)
        right_weight = np.zeros(n, dtype=np.float64)
        stddev = np.zeros(n, dtype=np.float64)

        self._n = n
        self._build_tree(
            bridge_index,
            left_index,
            right_index,
            left_weight,
            right_weight,
            stddev,
        )

        self._bridge_index = bridge_index
        self._left_index = left_index
        self._right_index = right_index
        self._left_weight = left_weight
        self._right_weight = right_weight
        self._stddev = stddev

    def _build_tree(
        self,
        bridge_index: NDArray[np.intp],
        left_index: NDArray[np.intp],
        right_index: NDArray[np.intp],
        left_weight: NDArray[np.float64],
        right_weight: NDArray[np.float64],
        stddev: NDArray[np.float64],
    ) -> None:
        """Build the binary-tree construction order.

        The algorithm uses a queue-based breadth-first bisection:
        1. First variate → terminal point W(T) = √T · Z[0]
        2. Remaining variates fill midpoints of progressively smaller
           intervals, conditioned on already-assigned endpoints.

        Reference: Jäckel (2002), Algorithm 10.1
        """
        n = self._n
        times = self._times

        bridge_index[0] = n
        stddev[0] = np.sqrt(times[n])
        left_index[0] = 0
        right_index[0] = 0
        left_weight[0] = 0.0
        right_weight[0] = 0.0

        if n == 1:
            return

        map_idx = np.zeros(n + 1, dtype=np.intp)
        map_idx[n] = 1

        queue: list[tuple[int, int]] = []

        queue.append((0, n))

        step = 1
        while queue and step < n:
            next_queue: list[tuple[int, int]] = []
            for left, right in queue:
                if right - left <= 1:
                    continue
                mid = (left + right) // 2
                bridge_index[step] = mid
                left_index[step] = left
                right_index[step] = right

                t_l = times[left]
                t_m = times[mid]
                t_r = times[right]

                dt_total = t_r - t_l

                if left == 0:
                    left_weight[step] = 0.0
                    right_weight[step] = (t_m - t_l) / dt_total
                    stddev[step] = np.sqrt((t_m - t_l) * (t_r - t_m) / dt_total)
                else:
                    left_weight[step] = (t_r - t_m) / dt_total
                    right_weight[step] = (t_m - t_l) / dt_total
                    stddev[step] = np.sqrt((t_m - t_l) * (t_r - t_m) / dt_total)

                map_idx[mid] = step + 1
                next_queue.append((left, mid))
                next_queue.append((mid, right))
                step += 1
                if step >= n:
                    break
            queue = next_queue

    @property
    def size(self) -> int:
        """Number of path increments (N = len(times) - 1)."""
        return self._n

    @property
    def times(self) -> NDArray[np.float64]:
        """The time grid used for construction."""
        return self._times.copy()

    def transform(self, normals: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Transform i.i.d. N(0,1) variates into Brownian path values.

        Parameters
        ----------
        normals : ndarray of shape (N,) or (N, d)
            Independent standard normal variates.  For d-dimensional
            Brownian motion, each column is an independent factor.

        Returns
        -------
        ndarray of shape (N+1,) or (N+1, d)
            Brownian motion path values W(t_0), W(t_1), ..., W(t_N)
            where W(t_0) = 0.  The joint distribution satisfies
            Cov(W(s), W(t)) = min(s, t) · I_d.
        """
        normals = np.asarray(normals, dtype=np.float64)
        multidim = normals.ndim == 2

        if multidim:
            if normals.shape[0] != self._n:
                raise ValueError(
                    f"Expected normals.shape[0] == {self._n}, got {normals.shape[0]}"
                )
            d = normals.shape[1]
            path: NDArray[np.float64] = np.zeros((self._n + 1, d), dtype=np.float64)
        else:
            if normals.shape[0] != self._n:
                raise ValueError(
                    f"Expected {self._n} normal variates, got {normals.shape[0]}"
                )
            path = np.zeros(self._n + 1, dtype=np.float64)

        path[self._bridge_index[0]] = self._stddev[0] * normals[0]

        for i in range(1, self._n):
            bi = self._bridge_index[i]
            li = self._left_index[i]
            ri = self._right_index[i]
            path[bi] = (
                self._left_weight[i] * path[li]
                + self._right_weight[i] * path[ri]
                + self._stddev[i] * normals[i]
            )

        return path

    def increments(self, normals: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Transform i.i.d. N(0,1) variates into Brownian increments.

        Parameters
        ----------
        normals : ndarray of shape (N,) or (N, d)
            Independent standard normal variates.

        Returns
        -------
        ndarray of shape (N,) or (N, d)
            Brownian increments ΔW_i = W(t_{i+1}) - W(t_i).
        """
        path = self.transform(normals)
        if path.ndim == 1:
            return np.diff(path)
        return np.diff(path, axis=0)
