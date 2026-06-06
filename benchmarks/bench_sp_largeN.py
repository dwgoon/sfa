# -*- coding: utf-8 -*-
"""SignalPropagation v0.1.0 vs v0.2.0 at large N where GEMV becomes
bandwidth-bound and CUDA can amortize launch overhead.

GPU memory note: an NxN fp32 matrix takes 4 * N**2 bytes; on a 24 GB
4090, the practical upper bound is around N = 76800 for a single fp32
W alone. We add a 1.5 GB safety margin and probe up to N = 49152 in
fp32 + N = 32768 in fp64.

CPU memory note: an NxN fp64 matrix at N = 49152 is 19.3 GB. Skip the
v0.1.0 fp64 baseline if the host does not have that much RAM.
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np

from _v010_baseline import propagate_iterative_v010
from sfa.algorithms.sp import SignalPropagation as SP_v020


def make_W(N: int, dtype, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Build the small W in fp64 then cast down to avoid double-precision
    # intermediates in the test inputs.
    W = (rng.standard_normal((N, N)) * (0.5 / np.sqrt(N))).astype(dtype)
    np.fill_diagonal(W, dtype(0.0))
    return W


def time_fn(fn, reps: int = 2) -> float:
    fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return min(samples)


def bench(N: int, reps: int):
    print(f"\n=== SignalPropagation, N={N} ===")
    print(f"   dense W footprint: fp64={N*N*8/1e9:5.1f} GB  "
          f"fp32={N*N*4/1e9:5.1f} GB")

    # We size the random vectors in fp64 for the comparison source,
    # then cast at the entry point of each backend.
    rng = np.random.default_rng(42)
    xi = np.zeros(N, dtype=np.float64)
    b  = (rng.standard_normal(N) * 0.1).astype(np.float64)
    a  = 0.5
    tol = 1e-7
    lim_iter = 200          # converges in ~13 iter for this W spectrum

    rows = []  # (label, seconds, vs_v010, max|diff|)

    # ---- v0.1.0 baseline (fp64) -----------------------------------------
    print("   building fp64 W ...", flush=True)
    W64 = make_W(N, np.float64)
    print("   v0.1.0 fp64 numpy iter ...", flush=True)
    t = time_fn(lambda: propagate_iterative_v010(
        W64, xi, b, a=a, lim_iter=lim_iter, tol=tol), reps=reps)
    x_ref, _ = propagate_iterative_v010(W64, xi, b, a=a,
                                          lim_iter=lim_iter, tol=tol)
    rows.append(("v0.1.0 numpy iter (fp64)", t, 1.0, 0.0))
    t_base = t

    # ---- v0.2.0 numpy iter, fp64 (preallocated) -------------------------
    alg = SP_v020("SP")
    print("   v0.2 fp64 numpy iter ...", flush=True)
    t = time_fn(lambda: alg.propagate_iterative(
        W64, xi, b, a=a, lim_iter=lim_iter, tol=tol,
        device="cpu", dtype=np.float64), reps=reps)
    x, _ = alg.propagate_iterative(W64, xi, b, a=a, lim_iter=lim_iter,
                                    tol=tol, device="cpu",
                                    dtype=np.float64)
    rows.append(("v0.2 numpy iter (fp64)", t, t_base / t,
                 float(np.abs(x - x_ref).max())))

    # We don't need W64 anymore; free it before allocating fp32 / GPU.
    del W64; gc.collect()

    # ---- v0.2.0 numpy iter, fp32 -----------------------------------------
    W32 = make_W(N, np.float32)
    print("   v0.2 fp32 numpy iter ...", flush=True)
    t = time_fn(lambda: alg.propagate_iterative(
        W32, xi, b, a=a, lim_iter=lim_iter, tol=tol,
        device="cpu", dtype=np.float32), reps=reps)
    x, _ = alg.propagate_iterative(W32, xi, b, a=a, lim_iter=lim_iter,
                                    tol=tol, device="cpu",
                                    dtype=np.float32)
    rows.append(("v0.2 numpy iter (fp32)", t, t_base / t,
                 float(np.abs(x.astype(np.float64) - x_ref).max())))

    # ---- v0.2.0 CUDA TF32 (fp32) -----------------------------------------
    try:
        from sfa import _cuda
        have_gpu = (_cuda.HAS_NATIVE
                    and _cuda.native_module().device_count() > 0)
    except Exception:
        have_gpu = False
    if have_gpu:
        print("   v0.2 CUDA TF32 ...", flush=True)
        t = time_fn(lambda: alg.propagate_iterative(
            W32, xi, b, a=a, lim_iter=lim_iter, tol=tol,
            device="cuda:0", dtype=np.float32), reps=reps)
        x, _ = alg.propagate_iterative(W32, xi, b, a=a,
                                        lim_iter=lim_iter, tol=tol,
                                        device="cuda:0",
                                        dtype=np.float32)
        rows.append(("v0.2 CUDA TF32 (fp32)", t, t_base / t,
                     float(np.abs(x.astype(np.float64) - x_ref).max())))

        # ---- CUDA fp16 (strict, opportunistic) --------------------------
        print("   v0.2 CUDA FP16 strict ...", flush=True)
        t = time_fn(lambda: alg.propagate_iterative(
            W32, xi, b, a=a, lim_iter=lim_iter, tol=tol,
            device="cuda:0", dtype=np.float16), reps=reps)
        x, _ = alg.propagate_iterative(W32, xi, b, a=a,
                                        lim_iter=lim_iter, tol=tol,
                                        device="cuda:0",
                                        dtype=np.float16)
        rows.append(("v0.2 CUDA FP16 strict ", t, t_base / t,
                     float(np.abs(x.astype(np.float64) - x_ref).max())))

    # ---- Report ----------------------------------------------------------
    print()
    print(f"   {'backend':<32} {'time (s)':>10} {'vs v0.1.0':>10} "
          f"{'max|diff|':>11}")
    print("   " + "-" * 68)
    for label, t, sp, err in rows:
        print(f"   {label:<32} {t:>10.4f} {sp:>9.2f}x {err:>11.2e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[16384, 24576, 32768, 40960])
    p.add_argument("--reps", type=int, default=2)
    args = p.parse_args()

    for N in args.sizes:
        try:
            bench(N, args.reps)
        except MemoryError as e:
            print(f"   N={N}: skipped ({e})")
        except RuntimeError as e:
            print(f"   N={N}: skipped ({e})")


if __name__ == "__main__":
    main()
