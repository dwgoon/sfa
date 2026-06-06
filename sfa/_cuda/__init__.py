# -*- coding: utf-8 -*-
"""Native CUDA kernels for sfa.

The native extension ``sfa._cuda._native`` is built from the C++/CUDA
sources under ``sfa/_cuda/src``. It is optional: if the extension is not
present (CUDA toolchain unavailable at build time, or build was skipped),
``HAS_NATIVE`` is False and callers should fall back to the cupy or numpy
implementations.
"""
from __future__ import annotations

import os
import sys


def _add_nvidia_dll_paths() -> None:
    """Make the bundled NVIDIA runtime DLLs locatable on Windows.

    The wheel declares ``nvidia-cublas-cu1X`` / ``nvidia-cuda-runtime-cu1X``
    as install_requires; those packages drop their DLLs under
    ``site-packages/nvidia/cublas/bin`` and ``cuda_runtime/bin`` on
    Windows. We register each of those directories with the Win32 DLL
    loader before importing the native extension so the dependent
    ``cublas64_X.dll`` / ``cudart64_X.dll`` are found regardless of how
    the user launched Python.

    On Linux / macOS the same problem is handled at link time via an
    ``$ORIGIN``-relative rpath embedded in the extension (see setup.py).
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    base = os.path.dirname(getattr(nvidia, "__file__", "") or "")
    if not base:
        return
    for sub in ("cublas/bin", "cuda_runtime/bin", "cuda_nvrtc/bin"):
        cand = os.path.join(base, *sub.split("/"))
        if os.path.isdir(cand):
            try:
                os.add_dll_directory(cand)  # type: ignore[attr-defined]
            except (AttributeError, OSError):
                # add_dll_directory is Python 3.8+ on Windows only; if
                # it is unavailable we silently fall back to PATH.
                pass


_add_nvidia_dll_paths()


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
