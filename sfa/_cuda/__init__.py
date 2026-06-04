# -*- coding: utf-8 -*-
"""Native CUDA kernels for sfa.

The native extension ``sfa._cuda._native`` is built from the C++/CUDA
sources under ``sfa/_cuda/src``. It is optional: if the extension is not
present (CUDA toolchain unavailable at build time, or build was skipped),
``HAS_NATIVE`` is False and callers should fall back to the cupy or numpy
implementations.
"""
from __future__ import annotations

# IMPORTANT: do not pre-create a module-level ``_native = None`` — that
# would shadow the submodule name and make ``from . import _native``
# silently bind ``None``. Import first, then assign fallbacks on failure.
try:
    from . import _native  # type: ignore[attr-defined]
    HAS_NATIVE = True
    _import_error: Exception | None = None
except Exception as _e:  # ImportError, OSError (missing DLL), etc.
    _native = None  # type: ignore[assignment]
    HAS_NATIVE = False
    _import_error = _e


def native_module():
    """Return the loaded native module, or raise a helpful error."""
    if not HAS_NATIVE:
        raise RuntimeError(
            "sfa native CUDA extension is not available. "
            "Build with `pip install -e .` inside the sfa-cu132 conda env "
            f"(see environment-cuda.yml). Underlying error: {_import_error!r}"
        )
    return _native


__all__ = ["HAS_NATIVE", "native_module"]
