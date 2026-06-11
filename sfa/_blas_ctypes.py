# -*- coding: utf-8 -*-
"""Runtime-loadable LAPACK dispatcher.

Some users have *both* MKL and OpenBLAS installed (e.g. via the ``mkl``
PyPI package or via ``conda install libblas=*=*mkl``). scipy is linked
against exactly one of them at install time, but we can still call the
other - or either - at runtime by loading the shared library with
ctypes and calling the LAPACKE C interface directly.

Public API
----------

``available_backends()``
    Return the list of canonical backend names we can actually load.

``solve(A, B, backend='auto', num_threads=None)``
    LAPACK ``?gesv`` via the requested backend. Falls back to scipy when
    backend='auto' and a more specific selection is not possible.

Supported backends
------------------
- ``'mkl'``       : Intel oneMKL ``libmkl_rt``.
- ``'openblas'``  : OpenBLAS ``libopenblas``.
- ``'auto'``      : pick whichever is loadable; prefer MKL.

The dispatcher caches the loaded handle per backend, so the first call
pays the dlopen cost and subsequent calls are essentially overhead-free.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# LAPACKE matrix layout constants (cblas.h).
_LAPACK_ROW_MAJOR = 101
_LAPACK_COL_MAJOR = 102


class BackendLoadError(RuntimeError):
    """Raised when the requested BLAS backend cannot be loaded."""


# ---------- Backend descriptors ---------------------------------------------

class _Backend:
    """Wraps one ctypes-loaded BLAS DLL."""

    def __init__(self, name: str, lib: ctypes.CDLL):
        self.name = name
        self.lib = lib

        # LAPACKE_dgesv / sgesv. Signatures match LAPACKE row-major:
        #   int LAPACKE_dgesv(int layout, int n, int nrhs,
        #                     double* a, int lda,
        #                     int* ipiv,
        #                     double* b, int ldb);
        self.dgesv = lib.LAPACKE_dgesv
        self.dgesv.restype = ctypes.c_int
        self.dgesv.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ]

        self.sgesv = lib.LAPACKE_sgesv
        self.sgesv.restype = ctypes.c_int
        self.sgesv.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]

        # Thread setters/getters are backend-specific.
        self.set_num_threads = self._resolve_thread_setter()
        self.get_num_threads = self._resolve_thread_getter()

    def _resolve_thread_getter(self):
        """Return a callable ``f() -> int`` reporting the backend thread count.

        Used to snapshot and restore the process-wide thread count around a
        single ``solve`` so a one-off ``num_threads`` does not leak into the
        rest of the process. Returns ``None`` when the backend exposes no
        getter (then the count is simply left as set).
        """
        lib = self.lib
        # MKL: int MKL_Get_Max_Threads(void) / int mkl_get_max_threads(void)
        for sym in ('MKL_Get_Max_Threads', 'mkl_get_max_threads'):
            fn = getattr(lib, sym, None)
            if fn is not None:
                fn.argtypes = []
                fn.restype = ctypes.c_int
                return fn
        # OpenBLAS: int openblas_get_num_threads(void) / goto_get_num_threads
        for sym in ('openblas_get_num_threads', 'goto_get_num_threads'):
            fn = getattr(lib, sym, None)
            if fn is not None:
                fn.argtypes = []
                fn.restype = ctypes.c_int
                return fn
        return None

    def _resolve_thread_setter(self):
        """Return a callable ``f(n: int) -> None`` for this backend."""
        lib = self.lib
        # MKL: void MKL_Set_Num_Threads(int n)
        # MKL: void mkl_set_num_threads(int n)
        for sym in ('MKL_Set_Num_Threads', 'mkl_set_num_threads'):
            fn = getattr(lib, sym, None)
            if fn is not None:
                fn.argtypes = [ctypes.c_int]
                fn.restype = None
                return fn
        # OpenBLAS: void openblas_set_num_threads(int n)
        # OpenBLAS: void goto_set_num_threads(int n)  (legacy alias)
        for sym in ('openblas_set_num_threads', 'goto_set_num_threads'):
            fn = getattr(lib, sym, None)
            if fn is not None:
                fn.argtypes = [ctypes.c_int]
                fn.restype = None
                return fn
        return lambda n: None  # no-op fallback


# ---------- Library probing --------------------------------------------------

def _conda_or_sys_prefix() -> Path:
    """Return CONDA_PREFIX if set, else sys.prefix.

    Inside a conda env CONDA_PREFIX matches sys.prefix; outside conda
    (or when a subshell forgot to forward the env var) we still want
    the active interpreter's prefix.
    """
    return Path(os.environ.get('CONDA_PREFIX', sys.prefix))


def _candidate_paths_mkl() -> list[str]:
    """Plausible locations of MKL on the host (Windows / Linux / macOS)."""
    paths = []

    # PyPI `mkl` package ships the redist DLL into the env's bin dir.
    try:
        import mkl  # noqa: F401
        mkl_dir = Path(sys.modules['mkl'].__file__).parent
        paths += [
            str(mkl_dir / 'mkl_rt.dll'),
            str(mkl_dir / 'libmkl_rt.so'),
            str(mkl_dir / 'libmkl_rt.so.2'),
        ]
    except ImportError:
        pass

    # conda / venv layout. We scan because MKL ships versioned filenames
    # (mkl_rt.1.dll, mkl_rt.2.dll, mkl_rt.3.dll, ...) and we cannot know
    # the SOABI version in advance.
    prefix = _conda_or_sys_prefix()
    if os.name == 'nt':
        bin_dir = prefix / 'Library' / 'bin'
        if bin_dir.is_dir():
            paths += sorted(
                (str(p) for p in bin_dir.glob('mkl_rt*.dll')),
                reverse=True)
    else:
        lib_dir = prefix / 'lib'
        if lib_dir.is_dir():
            paths += sorted(
                (str(p) for p in lib_dir.glob('libmkl_rt*.so*')),
                reverse=True)
            paths += sorted(
                (str(p) for p in lib_dir.glob('libmkl_rt*.dylib')),
                reverse=True)

    # Dynamic linker fallback.
    sysname = ctypes.util.find_library('mkl_rt')
    if sysname:
        paths.append(sysname)

    # Last-resort well-known names.
    paths += ['mkl_rt', 'mkl_rt.dll',
              'libmkl_rt.so', 'libmkl_rt.so.2',
              'libmkl_rt.dylib', 'libmkl_rt.1.dylib']
    # De-dup while preserving order.
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def _candidate_paths_openblas() -> list[str]:
    """Plausible locations of a usable LP64 OpenBLAS.

    NOTE: scipy 1.17+ vendors an *ILP64* OpenBLAS under names like
    ``libscipy_openblas64_-<hash>.dll`` with a non-standard symbol
    convention; the standard ``LAPACKE_dgesv`` C API is not exposed by
    that build, so it is intentionally excluded here. To use the ctypes
    runtime swap with OpenBLAS, install a stock LP64 OpenBLAS via conda
    (``conda install -c conda-forge libblas=*=*openblas``) or distribute
    a libopenblas alongside the package.
    """
    paths = []
    for mod_name in ('scipy_openblas32',):  # LP64 variant only
        try:
            mod = __import__(mod_name)
            getter = getattr(mod, 'get_library', None)
            if getter is not None:
                paths.append(getter())
        except ImportError:
            pass

    prefix = _conda_or_sys_prefix()
    if os.name == 'nt':
        bin_dir = prefix / 'Library' / 'bin'
        if bin_dir.is_dir():
            # Pick the non-`64_` variants explicitly (ILP64 = 64 in name).
            paths += [str(p) for p in bin_dir.glob('libopenblas*.dll')
                      if '64_' not in p.name]
    else:
        lib_dir = prefix / 'lib'
        if lib_dir.is_dir():
            paths += [str(p) for p in lib_dir.glob('libopenblas*')
                      if '64_' not in p.name]

    sysname = ctypes.util.find_library('openblas')
    if sysname:
        paths.append(sysname)

    paths += ['openblas', 'libopenblas.so', 'libopenblas.so.0',
              'libopenblas.dll', 'libopenblas.dylib']
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def _try_load(paths: list[str]) -> Optional[ctypes.CDLL]:
    for p in paths:
        try:
            return ctypes.CDLL(p)
        except OSError:
            continue
    return None


# ---------- Loader cache -----------------------------------------------------

_lock = threading.Lock()
_cache: dict[str, _Backend] = {}
_load_errors: dict[str, BackendLoadError] = {}


def _load(name: str) -> _Backend:
    name = name.lower()
    with _lock:
        if name in _cache:
            return _cache[name]
        if name in _load_errors:
            raise _load_errors[name]

        if name == 'mkl':
            lib = _try_load(_candidate_paths_mkl())
        elif name == 'openblas':
            lib = _try_load(_candidate_paths_openblas())
        else:
            err = BackendLoadError(f"Unknown BLAS backend: {name!r}")
            _load_errors[name] = err
            raise err

        if lib is None:
            err = BackendLoadError(
                f"Could not locate the {name!r} shared library. "
                "Install it (e.g. `pip install mkl` for MKL, "
                "`conda install -c conda-forge libblas=*=*openblas` for "
                "OpenBLAS) or place its .so/.dll on the loader path."
            )
            _load_errors[name] = err
            raise err

        try:
            backend = _Backend(name, lib)
        except AttributeError as e:
            err = BackendLoadError(
                f"The {name!r} library was loaded but is missing the "
                f"required LAPACKE symbol: {e}. The library is probably "
                "an older or stripped build."
            )
            _load_errors[name] = err
            raise err

        _cache[name] = backend
        return backend


def available_backends() -> list[str]:
    """Return canonical names of backends we can currently load by ctypes."""
    out = []
    for name in ('mkl', 'openblas'):
        try:
            _load(name)
            out.append(name)
        except BackendLoadError:
            continue
    return out


# ---------- High-level solver --------------------------------------------

def solve(A: np.ndarray, B: np.ndarray, *,
          backend: str = 'auto',
          num_threads: Optional[int] = None) -> np.ndarray:
    """Solve A X = B via LAPACK ``?gesv`` on the requested backend.

    A and B must be 2D and ``float32`` or ``float64``. A is modified in
    place (LU factors); B is overwritten with X.

    Parameters
    ----------
    backend
        ``'mkl'``, ``'openblas'``, or ``'auto'`` (prefer MKL, fall back
        to OpenBLAS).
    num_threads
        If given, set the backend's thread count before the solve.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D.")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("B must be (N, nrhs) with N == A.shape[0].")
    if A.dtype != B.dtype:
        raise ValueError(f"A.dtype ({A.dtype}) != B.dtype ({B.dtype}).")
    if A.dtype not in (np.float32, np.float64):
        raise ValueError("Only float32 and float64 are supported here.")

    canon = backend.lower()
    if canon in ('auto', 'default', ''):
        try:
            be = _load('mkl')
        except BackendLoadError:
            be = _load('openblas')   # last resort
    else:
        be = _load(canon)

    # Snapshot the backend's current thread count so a one-off num_threads
    # does not leak into the rest of the process; restore it after the solve.
    prev_threads = None
    if num_threads is not None:
        if be.get_num_threads is not None:
            try:
                prev_threads = int(be.get_num_threads())
            except Exception:
                prev_threads = None
        be.set_num_threads(int(num_threads))

    # Make sure layout is C-contiguous (LAPACK_ROW_MAJOR friendly).
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    N = A.shape[0]
    NRHS = B.shape[1]
    ipiv = np.empty(N, dtype=np.int32)

    try:
        if A.dtype == np.float64:
            info = be.dgesv(
                _LAPACK_ROW_MAJOR, N, NRHS,
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N,
                ipiv.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), NRHS,
            )
        else:  # float32
            info = be.sgesv(
                _LAPACK_ROW_MAJOR, N, NRHS,
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
                ipiv.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), NRHS,
            )
    finally:
        if prev_threads is not None:
            be.set_num_threads(prev_threads)

    if info != 0:
        raise np.linalg.LinAlgError(
            f"LAPACKE_?gesv on backend {be.name!r} returned info={info}.")
    return B


__all__ = ["available_backends", "solve", "BackendLoadError"]
