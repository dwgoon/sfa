# -*- coding: utf-8 -*-
"""compute_influence CPU LAPACK fast path across thread counts and
BLAS backends (scipy-linked OpenBLAS vs. runtime-loaded MKL via ctypes).
"""
import time

import numpy as np

import sfa
from sfa.control.influence import compute_influence
from sfa._blas_ctypes import available_backends


def make_W(N, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((N, N)) * (0.5 / np.sqrt(N))).astype(np.float64)


def time_call(fn, reps=3):
    fn()  # warm-up
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return min(samples)


def main():
    N = 4096
    print(f"N={N}, dtype=float64, closed-form LAPACK ?gesv\n")
    print(f"scipy-linked BLAS: {sfa.blas.loaded_backends()}")
    print(f"ctypes-loadable  : {available_backends()}\n")

    W = make_W(N)
    np.fill_diagonal(W, 0.0)

    # Discover candidate backends + thread counts to sweep.
    backends = [None]  # scipy default (whatever is linked)
    backends += available_backends()  # any ctypes-loadable extras

    threads = [1, 2, 4, 8, 16, 24]

    print(f"{'threads':>8} " + " ".join(
        f"{('scipy' if b is None else b):>12}" for b in backends))
    print("-" * (9 + 13 * len(backends)))

    for n in threads:
        row = [f"{n:>8d}"]
        for b in backends:
            t = time_call(lambda b=b, n=n: compute_influence(
                W, alpha=0.5, beta=0.5, rtype="array", device="cpu",
                dtype=np.float64, num_threads=n, backend=b))
            row.append(f"{1000*t:>9.1f}ms")
        print(" ".join(row))

    # Verify the MKL-backed result matches scipy-backed.
    print("\n--- correctness check (N=1024) ---")
    W2 = make_W(1024)
    np.fill_diagonal(W2, 0.0)
    S_scipy = compute_influence(W2, alpha=0.5, beta=0.5, rtype="array",
                                device="cpu", dtype=np.float64, backend=None)
    for b in available_backends():
        S = compute_influence(W2, alpha=0.5, beta=0.5, rtype="array",
                              device="cpu", dtype=np.float64, backend=b)
        diff = float(np.abs(S - S_scipy).max())
        print(f"  scipy vs {b}: max|diff| = {diff:.2e}")


if __name__ == "__main__":
    main()
