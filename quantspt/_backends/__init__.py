"""Compute backend abstraction.

Provides a uniform array API across NumPy (default), Numba JIT,
JAX (GPU + autodiff), and CuPy (raw CUDA). The active backend is
selected via :func:`quantspt.set_backend`.
"""

__all__: list[str] = []
