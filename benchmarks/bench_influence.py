# -*- coding: utf-8 -*-
"""Benchmark the influence backends.

Backends:
    cpu      : numpy (float64 internally)
    gpu      : cupy GPU path in sfa.control.influence._compute_influence_gpu
    cuda     : native CUDA extension (dtype-controllable)

Usage (from inside the sfa-cu132 conda env):

    python benchmarks/bench_influence.py --sizes 512 2048 4096 --reps 3 \\
        --backends cpu gpu cuda \\
        --dtypes float32 float64 float16

For each (N, backend [, dtype]) cell we report best and median wall time
over `--reps` repetitions, plus the iteration count of the last run.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _make_W(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n)).astype(np.float32) * (0.5 / np.sqrt(n))
    np.fill_diagonal(W, 0.0)
    return W


def _time(fn, reps: int) -> tuple[float, float]:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    samples.sort()
    return samples[0], samples[len(samples) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--backends", nargs="+",
                   default=["cpu", "gpu", "cuda"],
                   help="cpu=numpy, gpu=cupy, cuda=native")
    p.add_argument("--dtypes", nargs="+",
                   default=["float32"],
                   help="dtypes for the cuda backend (ignored for cpu/gpu)")
    args = p.parse_args()

    from sfa.control.influence import compute_influence

    have_cupy = False
    try:
        import cupy  # noqa: F401
        have_cupy = True
    except ImportError:
        pass

    have_native = False
    try:
        from sfa import _cuda
        have_native = _cuda.HAS_NATIVE
    except ImportError:
        pass

    headers = ["N", "backend", "dtype", "best (s)", "median (s)", "iters"]
    fmt = "{:>6} | {:>10} | {:>9} | {:>10} | {:>11} | {:>6}"
    print(fmt.format(*headers))
    print("-" * 72)

    for N in args.sizes:
        W = _make_W(N)
        # cuBLAS cuBLAS warm-up on first iter is unavoidable; we let `_time`
        # include the warm-up sample in the worst case (we keep best/median).

        for backend in args.backends:
            if backend == "gpu" and not have_cupy:
                continue
            if backend == "cuda" and not have_native:
                continue

            if backend == "cuda":
                dtypes = args.dtypes
            else:
                dtypes = ["-"]

            for dt in dtypes:
                dev = {"cpu": "cpu", "gpu": "gpu:0", "cuda": "cuda:0"}[backend]
                kwargs = dict(
                    alpha=args.alpha, beta=args.beta, rtype="array",
                    max_iter=args.max_iter, tol=args.tol,
                    device=dev, get_iter=True)
                if backend == "cuda":
                    kwargs["dtype"] = np.dtype(dt)

                def run():
                    return compute_influence(W, **kwargs)

                # priming run
                _, num_iter = run()
                best, med = _time(lambda: run(), args.reps)
                print(fmt.format(
                    N, backend, dt, f"{best:.4f}", f"{med:.4f}", num_iter))


if __name__ == "__main__":
    main()
