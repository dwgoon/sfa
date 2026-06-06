# -*- coding: utf-8 -*-
"""Shared pytest skip predicates for the sfa test suite.

Two distinct conditions matter when deciding whether to run a CUDA test:

1. ``HAS_NATIVE`` - the C++/CUDA extension was successfully built and
   imported. Without it, no CUDA code path exists in this process.
2. ``device_count() > 0`` - the runtime sees at least one NVIDIA GPU.
   On a CUDA-enabled CI wheel runner without a GPU, the extension loads
   fine but ``cudaGetDeviceCount`` returns 0, and any actual CUDA call
   will fail.

``no_cuda_reason`` returns a non-None string when CUDA tests should be
skipped, suitable for ``@pytest.mark.skipif``.
"""
from __future__ import annotations

import pytest


def _native_status():
    try:
        from sfa import _cuda
    except Exception:
        return False, None
    if not _cuda.HAS_NATIVE:
        return False, None
    try:
        return True, _cuda.native_module()
    except Exception:
        return False, None


def no_cuda_reason() -> str | None:
    """Return a non-None skip reason if CUDA tests should not run, else None."""
    ok, native = _native_status()
    if not ok:
        return "sfa native CUDA extension not built"
    try:
        n = native.device_count()
    except Exception as e:
        return f"CUDA driver unavailable: {e!r}"
    if n <= 0:
        return "no NVIDIA GPU visible to the runtime"
    return None


# Convenience decorator: @requires_cuda above any cuda-using test.
requires_cuda = pytest.mark.skipif(
    no_cuda_reason() is not None,
    reason=no_cuda_reason() or "cuda available",
)
