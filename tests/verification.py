# -*- coding: utf-8 -*-
"""Portable post-install verification for sfa.

Runs without pytest. Exits 0 on success, non-zero on failure. Designed to
be runnable from any environment that has sfa installed:

    python -m tests.verification
    # or, if `tests/` is not on sys.path:
    python path/to/tests/verification.py

What it checks
--------------
1. ``import sfa`` and version string is present.
2. ``import sfa._cuda`` succeeds (even on CPU-only wheels — the package
   provides a graceful fallback).
3. CPU influence computation produces a plausible result (shape, dtype,
   finite values).
4. CPU SignalPropagation propagate_iterative produces a plausible result
   and a trajectory of the expected shape.
5. If a CUDA GPU is visible, a quick GPU verification (tiny influence)
   also runs.

The CUDA portion is purely opportunistic - the script does NOT fail when
no GPU is present. That way the same verification script works on plain
CPU installs, CUDA installs without a GPU on the build runner, and full
CUDA installs on developer machines.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np


def _step(name: str):
    print(f"-- {name}")


def main() -> int:
    # ------------------------------------------------------------------
    _step("import sfa")
    import sfa
    print(f"   sfa version: {sfa.__version__}")

    # ------------------------------------------------------------------
    _step("import sfa._cuda")
    from sfa import _cuda
    has_native = bool(_cuda.HAS_NATIVE)
    print(f"   HAS_NATIVE: {has_native}")
    if has_native:
        # If the native extension is present, check the GPU situation
        # without failing on absence.
        try:
            m = _cuda.native_module()
            n = m.device_count()
            print(f"   CUDA device_count: {n}")
            if n > 0:
                print(f"   device_info(0): {m.device_info(0)}")
        except Exception as e:
            print(f"   CUDA runtime call raised (ignored): {e!r}")

    # ------------------------------------------------------------------
    _step("CPU influence verification (LAPACK closed-form)")
    from sfa.control.influence import compute_influence
    N = 32
    rng = np.random.default_rng(0)
    # Cast after the scalar multiplication so the final dtype stays float32
    # (numpy would otherwise upcast through the Python-double scale factor).
    W = (rng.standard_normal((N, N)) * (0.5 / np.sqrt(N))).astype(np.float32)
    np.fill_diagonal(W, 0.0)
    S = compute_influence(W, alpha=0.5, beta=0.5, device="cpu", rtype="array")
    assert S.shape == (N, N), f"unexpected S.shape {S.shape}"
    assert S.dtype == np.float32, f"unexpected S.dtype {S.dtype}"
    assert np.isfinite(S).all(), "S contains non-finite values"
    print(f"   ||S||_F={np.linalg.norm(S):.4f}  dtype={S.dtype}")

    # ------------------------------------------------------------------
    _step("CPU SignalPropagation verification (trajectory)")
    from sfa.algorithms.sp import SignalPropagation
    alg = SignalPropagation("SP")
    xi = np.zeros(N, dtype=np.float64)
    b  = rng.standard_normal(N) * 0.1
    x, trj = alg.propagate_iterative(
        W.astype(np.float64), xi, b, a=0.5,
        lim_iter=200, tol=1e-7, get_trj=True, device="cpu")
    assert trj.ndim == 2 and trj.shape[1] == N
    assert np.allclose(trj[0], xi), "trajectory[0] should be xi"
    assert np.allclose(trj[-1], x), "trajectory[-1] should be x_final"
    print(f"   iters={trj.shape[0]-1}  ||x||={np.linalg.norm(x):.4f}")

    # ------------------------------------------------------------------
    # Opportunistic GPU verification
    if has_native:
        try:
            m = _cuda.native_module()
            if m.device_count() > 0:
                _step("CUDA influence verification (opportunistic)")
                S_gpu = compute_influence(
                    W, alpha=0.5, beta=0.5, device="cuda:0",
                    rtype="array", dtype=np.float32)
                assert S_gpu.shape == (N, N)
                print(f"   ||S_gpu||_F={np.linalg.norm(S_gpu):.4f}")
        except Exception as e:
            print(f"   GPU verification skipped: {e!r}")

    print("ALL OK")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
