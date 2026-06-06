# -*- coding: utf-8 -*-
"""Benchmark v0.1.0 reference vs v0.2.0.dev0 implementations.

Two workloads:
  1. ``compute_influence`` on dense synthetic W.
  2. ``SignalPropagation.propagate_iterative`` on the same W, with and
     without trajectory recording.

Backends compared
-----------------
compute_influence:
  * v0.1.0 CPU iterative                     (numpy DGEMM loop)
  * v0.2.0 CPU LAPACK closed-form (float64)  (scipy.linalg.solve)
  * v0.2.0 CPU LAPACK closed-form (float32)  (SGESV)
  * v0.2.0 CUDA float32 TF32                 (native)
  * v0.2.0 CUDA float32 (no TF32)            (native)
  * v0.2.0 CUDA float16                      (native, strict HMMA fp16)

SignalPropagation:
  * v0.1.0 numpy iterative (no preallocation)
  * v0.2.0 numpy iterative (preallocated trajectory)
  * v0.2.0 CUDA float32 TF32                 (native)

Each cell reports best-of-3 wallclock time after one warm-up call.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from _v010_baseline import (
    compute_influence_cpu_v010,
    propagate_iterative_v010,
)

# v0.2.0 entry points
from sfa.control.influence import compute_influence as compute_influence_v020
from sfa.algorithms.sp import SignalPropagation as SP_v020


def make_W(N: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Spectral-radius-safe random weights: ~|W|*alpha < 1 for alpha=0.5.
    W = rng.standard_normal((N, N)).astype(np.float64) * (0.5 / np.sqrt(N))
    np.fill_diagonal(W, 0.0)
    return W


def make_xb(N: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    xi = np.zeros(N, dtype=np.float64)
    b = rng.standard_normal(N) * 0.1
    return xi, b


def time_fn(fn, reps: int = 5) -> tuple[float, float, float]:
    """Wallclock stats after one warm-up; returns (mean, stddev, best)."""
    fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    arr = np.array(samples)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min())


@dataclass
class Row:
    label: str
    mean: float
    stddev: float
    best: float
    speedup: float   # vs baseline, computed on mean
    err: float


def fmt(rows: list[Row], baseline: str) -> str:
    lines = [
        f"{'backend':<32} {'mean (s)':>10} {'+/- std':>10} "
        f"{'vs ' + baseline:>10} {'max|diff|':>11}"
    ]
    lines.append("-" * 80)
    for r in rows:
        lines.append(f"{r.label:<32} {r.mean:>10.4f} {r.stddev:>10.4f} "
                     f"{r.speedup:>9.2f}x {r.err:>11.2e}")
    return "\n".join(lines)


def bench_influence(N: int, reps: int):
    print(f"\n=== compute_influence, N={N} ===")
    W = make_W(N)
    alpha, beta = 0.5, 0.5

    # Baseline (v0.1.0 CPU iterative)
    mean_b, std_b, best_b = time_fn(lambda: compute_influence_cpu_v010(
        W, alpha=alpha, beta=beta), reps=reps)
    S_base = compute_influence_cpu_v010(W, alpha=alpha, beta=beta)
    rows = [Row("v0.1.0 CPU iter (float64)",
                mean_b, std_b, best_b, 1.0, 0.0)]

    # v0.2.0 CPU LAPACK closed form, float64
    mean, std, best = time_fn(lambda: compute_influence_v020(
        W, alpha=alpha, beta=beta, rtype="array",
        device="cpu", dtype=np.float64), reps=reps)
    S = compute_influence_v020(W, alpha=alpha, beta=beta, rtype="array",
                               device="cpu", dtype=np.float64)
    rows.append(Row("v0.2 CPU LAPACK (float64)",
                    mean, std, best, mean_b / mean,
                    float(np.abs(S - S_base).max())))

    # v0.2.0 CPU LAPACK closed form, float32
    Wf32 = W.astype(np.float32)
    mean, std, best = time_fn(lambda: compute_influence_v020(
        Wf32, alpha=alpha, beta=beta, rtype="array",
        device="cpu", dtype=np.float32), reps=reps)
    S = compute_influence_v020(Wf32, alpha=alpha, beta=beta, rtype="array",
                               device="cpu", dtype=np.float32)
    rows.append(Row("v0.2 CPU LAPACK (float32)",
                    mean, std, best, mean_b / mean,
                    float(np.abs(S.astype(np.float64) - S_base).max())))

    # v0.2.0 CUDA paths (skipped if native CUDA unavailable / no GPU)
    try:
        from sfa import _cuda
        have_gpu = (_cuda.HAS_NATIVE
                    and _cuda.native_module().device_count() > 0)
    except Exception:
        have_gpu = False

    if have_gpu:
        for label, dtype, use_tf32 in (
                ("v0.2 CUDA (float64)",           np.float64, True),
                ("v0.2 CUDA TF32 (float32)",      np.float32, True),
                ("v0.2 CUDA FP32 (no TF32)",      np.float32, False),
                ("v0.2 CUDA FP16 strict",         np.float16, True),
        ):
            mean, std, best = time_fn(
                lambda d=dtype, u=use_tf32: compute_influence_v020(
                    W, alpha=alpha, beta=beta, rtype="array",
                    device="cuda:0", dtype=d, use_tf32=u), reps=reps)
            S = compute_influence_v020(W, alpha=alpha, beta=beta,
                                       rtype="array", device="cuda:0",
                                       dtype=dtype, use_tf32=use_tf32)
            err = float(np.abs(S.astype(np.float64) - S_base).max())
            rows.append(Row(label, mean, std, best, mean_b / mean, err))

    print(fmt(rows, "v0.1.0"))


def bench_signal_propagation(N: int, reps: int):
    print(f"\n=== SignalPropagation.propagate_iterative, N={N} (trj=False) ===")
    W = make_W(N)
    xi, b = make_xb(N)
    a = 0.5
    tol = 1e-7
    lim_iter = 2000

    # v0.1.0 baseline
    mean_b, std_b, best_b = time_fn(lambda: propagate_iterative_v010(
        W, xi, b, a=a, lim_iter=lim_iter, tol=tol), reps=reps)
    x_base, _ = propagate_iterative_v010(W, xi, b, a=a, lim_iter=lim_iter,
                                          tol=tol)
    rows = [Row("v0.1.0 numpy iter", mean_b, std_b, best_b, 1.0, 0.0)]

    # v0.2.0 numpy iter (pre-allocated trj buffer, but get_trj=False)
    alg = SP_v020("SP")
    mean, std, best = time_fn(lambda: alg.propagate_iterative(
        W, xi, b, a=a, lim_iter=lim_iter, tol=tol, device="cpu",
        dtype=np.float64), reps=reps)
    x, _ = alg.propagate_iterative(W, xi, b, a=a, lim_iter=lim_iter,
                                    tol=tol, device="cpu", dtype=np.float64)
    rows.append(Row("v0.2 numpy iter (preallocated)",
                    mean, std, best, mean_b / mean,
                    float(np.abs(x - x_base).max())))

    # v0.2.0 CUDA TF32
    try:
        from sfa import _cuda
        have_gpu = (_cuda.HAS_NATIVE
                    and _cuda.native_module().device_count() > 0)
    except Exception:
        have_gpu = False
    if have_gpu:
        mean, std, best = time_fn(lambda: alg.propagate_iterative(
            W, xi, b, a=a, lim_iter=lim_iter, tol=tol,
            device="cuda:0", dtype=np.float32), reps=reps)
        x, _ = alg.propagate_iterative(W, xi, b, a=a, lim_iter=lim_iter,
                                        tol=tol, device="cuda:0",
                                        dtype=np.float32)
        rows.append(Row("v0.2 CUDA TF32 (float32)",
                        mean, std, best, mean_b / mean,
                        float(np.abs(x.astype(np.float64) - x_base).max())))

    print(fmt(rows, "v0.1.0"))

    # ---------------------------------------------------------------- trj=True
    print(f"\n=== SignalPropagation.propagate_iterative, N={N} (trj=True) ===")
    mean_b, std_b, best_b = time_fn(lambda: propagate_iterative_v010(
        W, xi, b, a=a, lim_iter=lim_iter, tol=tol, get_trj=True),
        reps=reps)
    _, trj_base = propagate_iterative_v010(W, xi, b, a=a,
                                            lim_iter=lim_iter, tol=tol,
                                            get_trj=True)
    rows = [Row("v0.1.0 numpy iter + list_trj",
                mean_b, std_b, best_b, 1.0, 0.0)]

    mean, std, best = time_fn(lambda: alg.propagate_iterative(
        W, xi, b, a=a, lim_iter=lim_iter, tol=tol, get_trj=True,
        device="cpu", dtype=np.float64), reps=reps)
    _, trj = alg.propagate_iterative(W, xi, b, a=a, lim_iter=lim_iter,
                                      tol=tol, get_trj=True,
                                      device="cpu", dtype=np.float64)
    err = float(np.abs(trj[:trj_base.shape[0]] - trj_base).max()) \
        if trj.shape[0] >= trj_base.shape[0] else float("nan")
    rows.append(Row("v0.2 numpy iter (preallocated)",
                    mean, std, best, mean_b / mean, err))

    if have_gpu:
        mean, std, best = time_fn(lambda: alg.propagate_iterative(
            W, xi, b, a=a, lim_iter=lim_iter, tol=tol, get_trj=True,
            device="cuda:0", dtype=np.float32), reps=reps)
        _, trj = alg.propagate_iterative(W, xi, b, a=a, lim_iter=lim_iter,
                                          tol=tol, get_trj=True,
                                          device="cuda:0", dtype=np.float32)
        n_common = min(trj.shape[0], trj_base.shape[0])
        err = float(np.abs(trj[:n_common].astype(np.float64)
                           - trj_base[:n_common]).max())
        rows.append(Row("v0.2 CUDA TF32 (float32)",
                        mean, std, best, mean_b / mean, err))

    print(fmt(rows, "v0.1.0"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[256, 512, 1024, 2048, 4096])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--no-influence", action="store_true")
    p.add_argument("--no-sp", action="store_true")
    p.add_argument("--sp-sizes", nargs="+", type=int,
                   default=[512, 2048, 8192])
    args = p.parse_args()

    if not args.no_influence:
        for N in args.sizes:
            bench_influence(N, reps=args.reps)

    if not args.no_sp:
        for N in args.sp_sizes:
            bench_signal_propagation(N, reps=args.reps)


if __name__ == "__main__":
    main()
