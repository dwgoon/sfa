# -*- coding: utf-8 -*-
"""CPU BLAS introspection + thread control for sfa.

This module exposes a thin façade over `threadpoolctl` so the rest of
sfa can:

1. Discover which BLAS implementation (OpenBLAS / MKL / BLIS / Apple
   Accelerate) is currently loaded by numpy/scipy.
2. Temporarily restrict the BLAS thread count for a block of code
   (used by ``compute_influence`` and ``propagate_iterative`` when the
   user passes ``num_threads``).

For runtime *backend* switching (loading a different BLAS .dll/.so by
ctypes and calling it directly, bypassing scipy), see
``sfa._blas_ctypes``. Both modules share the canonical backend names:

    'mkl', 'openblas', 'blis', 'accelerate'
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional

try:
    from threadpoolctl import threadpool_info, threadpool_limits
    HAS_THREADPOOLCTL = True
except ImportError:
    HAS_THREADPOOLCTL = False


# Normalisation table: threadpoolctl's `internal_api` -> our canonical name.
_API_TO_NAME = {
    'mkl': 'mkl',
    'mkl_rt': 'mkl',
    'openblas': 'openblas',
    'blis': 'blis',
    'accelerate': 'accelerate',
}


def info() -> list[dict]:
    """Return a list of BLAS records currently loaded into this process."""
    if not HAS_THREADPOOLCTL:
        return []
    return [r for r in threadpool_info() if r.get('user_api') == 'blas']


def loaded_backends() -> list[str]:
    """Return canonical names of BLAS backends loaded in this process."""
    seen = []
    for r in info():
        n = _API_TO_NAME.get(r.get('internal_api', '').lower())
        if n and n not in seen:
            seen.append(n)
    return seen


def num_threads() -> Optional[int]:
    """Return the current BLAS thread count, or None if unknown."""
    recs = info()
    if not recs:
        return None
    # All records should agree if we set threads through threadpoolctl.
    return int(recs[0].get('num_threads', 0)) or None


def _canon(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    n = name.lower()
    if n in ('auto', 'default', ''):
        return None
    return _API_TO_NAME.get(n, n)


@contextmanager
def thread_limit(num_threads: Optional[int] = None):
    """Restrict the loaded BLAS to ``num_threads`` for a block of code.

    Wraps ``threadpoolctl.threadpool_limits(user_api='blas')`` and is a
    no-op when ``num_threads`` is ``None`` or when ``threadpoolctl`` is
    not installed.
    """
    if num_threads is None or not HAS_THREADPOOLCTL:
        yield
        return
    with threadpool_limits(limits=int(num_threads), user_api='blas'):
        yield


__all__ = [
    "info", "loaded_backends", "num_threads", "thread_limit",
    "HAS_THREADPOOLCTL",
]
