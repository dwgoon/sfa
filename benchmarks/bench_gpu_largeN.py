# -*- coding: utf-8 -*-
"""GPU-only `compute_influence` benchmark at large N.

The v0.1.0 CPU iterative baseline becomes impractical past N ~ 4096
(O(N^3) per iterate, many seconds to minutes). At that regime we just
want to see how the GPU paths scale among themselves and how the
LAPACK CPU fast path holds up.

Backends compared
-----------------
- v0.2.0 CPU LAPACK closed-form (fp64)   (scipy.linalg.solve)
- v0.2.0 CUDA TF32       (fp32)
- v0.2.0 CUDA FP32       (no TF32)
- v0.2.0 CUDA FP16
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np

from sfa.control.influence import compute_influence


def make_W(N: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, N)).astype(np.float64) * (0.5 / np.sqrt(N))
    np.fill_diagonal(W, 0.0)
    return W


def time_fn(fn, reps: int = 5) -> tuple[float, float, float]:
    """Wallclock stats after one warm-up; returns (mean, stddev, best)."""
    fn()  # warm-up
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    arr = np.array(samples)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min())


def fmt_time(t: float) -> str:
    if t < 1.0:
        return f"{t*1000:.0f} ms"
    return f"{t:.2f} s"


def bench_influence(N: int, reps: int, skip_cpu: bool):
    print(f"\n=== compute_influence, N={N} ===")
    print(f"   fp32 W footprint: {N*N*4/1e9:.2f} GB")
    W = make_W(N)
    alpha, beta = 0.5, 0.5
    rows = []  # (label, mean, stddev, best, max|diff| vs CPU fp64)

    # ---- v0.2 CPU LAPACK fp64 (reference) -------------------------------
    if not skip_cpu:
        print("   v0.2 CPU LAPACK (fp64) ...", flush=True)
        mean, std, best = time_fn(lambda: compute_influence(
            W, alpha=alpha, beta=beta, rtype="array",
            device="cpu", dtype=np.float64), reps=reps)
        S_ref = compute_influence(W, alpha=alpha, beta=beta, rtype="array",
                                  device="cpu", dtype=np.float64)
        rows.append(("v0.2 CPU LAPACK (fp64)", mean, std, best, 0.0))
    else:
        S_ref = None

    # ---- v0.2 CUDA TF32 fp32 --------------------------------------------
    print("   v0.2 CUDA TF32 (fp32) ...", flush=True)
    mean, std, best = time_fn(lambda: compute_influence(
        W, alpha=alpha, beta=beta, rtype="array",
        device="cuda:0", dtype=np.float32, use_tf32=True), reps=reps)
    S = compute_influence(W, alpha=alpha, beta=beta, rtype="array",
                          device="cuda:0", dtype=np.float32, use_tf32=True)
    err = (float(np.abs(S.astype(np.float64) - S_ref).max())
           if S_ref is not None else float("nan"))
    rows.append(("v0.2 CUDA TF32 (fp32)", mean, std, best, err))

    # ---- v0.2 CUDA FP32, no TF32 ----------------------------------------
    print("   v0.2 CUDA FP32 (no TF32) ...", flush=True)
    mean, std, best = time_fn(lambda: compute_influence(
        W, alpha=alpha, beta=beta, rtype="array",
        device="cuda:0", dtype=np.float32, use_tf32=False), reps=reps)
    S = compute_influence(W, alpha=alpha, beta=beta, rtype="array",
                          device="cuda:0", dtype=np.float32, use_tf32=False)
    err = (float(np.abs(S.astype(np.float64) - S_ref).max())
           if S_ref is not None else float("nan"))
    rows.append(("v0.2 CUDA FP32 (no TF32)", mean, std, best, err))

    # ---- v0.2 CUDA FP16 -------------------------------------------------
    print("   v0.2 CUDA FP16 ...", flush=True)
    mean, std, best = time_fn(lambda: compute_influence(
        W, alpha=alpha, beta=beta, rtype="array",
        device="cuda:0", dtype=np.float16, use_tf32=True), reps=reps)
    S = compute_influence(W, alpha=alpha, beta=beta, rtype="array",
                          device="cuda:0", dtype=np.float16, use_tf32=True)
    err = (float(np.abs(S.astype(np.float64) - S_ref).max())
           if S_ref is not None else float("nan"))
    rows.append(("v0.2 CUDA FP16", mean, std, best, err))

    # ---- Report ---------------------------------------------------------
    print()
    print(f"   {'backend':<28} {'mean':>10} {'+/- std':>10}   "
          f"{'max|diff| vs fp64':>18}")
    print("   " + "-" * 72)
    for label, mean, std, best, err in rows:
        err_s = "-" if np.isnan(err) else f"{err:.2e}"
        print(f"   {label:<28} {fmt_time(mean):>10} "
              f"{fmt_time(std):>10}   {err_s:>18}")

    del W; gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[5000, 10000, 20000])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--skip-cpu", action="store_true",
                   help="Skip the CPU LAPACK fp64 baseline (saves wallclock "
                        "at very large N; max|diff| then becomes unavailable).")
    args = p.parse_args()

    # Sanity: ensure a GPU is actually present.
    from sfa import _cuda
    if not (_cuda.HAS_NATIVE
            and _cuda.native_module().device_count() > 0):
        raise SystemExit("No CUDA device visible; this script is GPU-only.")

    for N in args.sizes:
        try:
            bench_influence(N, args.reps, args.skip_cpu)
        except (MemoryError, RuntimeError) as e:
            print(f"   N={N}: skipped ({e})")


if __name__ == "__main__":
    main()
