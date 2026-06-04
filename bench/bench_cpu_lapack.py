"""Compare CPU iterative path (old) vs LAPACK closed-form (new fast path)."""
import time
import numpy as np
from sfa.control.influence import compute_influence


def make_W(N, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, N)) * (0.5 / np.sqrt(N))
    np.fill_diagonal(W, 0.0)
    return W


def time_fn(fn, reps=3):
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    samples.sort()
    return samples[0]


for N in [512, 1024, 2048, 4096]:
    W = make_W(N)
    # Iterative (forced by get_iter=True)
    fn_iter = lambda: compute_influence(
        W, alpha=0.5, beta=0.5, rtype="array",
        device="cpu", get_iter=True)
    # Closed form (default when S=None and get_iter=False)
    fn_solve_f64 = lambda: compute_influence(
        W, alpha=0.5, beta=0.5, rtype="array",
        device="cpu", dtype=np.float64)
    fn_solve_f32 = lambda: compute_influence(
        W, alpha=0.5, beta=0.5, rtype="array",
        device="cpu", dtype=np.float32)

    t_iter = time_fn(fn_iter)
    t_solve_f64 = time_fn(fn_solve_f64)
    t_solve_f32 = time_fn(fn_solve_f32)
    print(f"N={N:>5}  iter(f64)={t_iter:7.3f}s   "
          f"solve(f64)={t_solve_f64:7.3f}s ({t_iter/t_solve_f64:5.2f}x)  "
          f"solve(f32)={t_solve_f32:7.3f}s ({t_iter/t_solve_f32:5.2f}x)")
