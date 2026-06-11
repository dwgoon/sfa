"""CPU dtype + LAPACK fast path; CUDA trajectory parity with CPU."""
from __future__ import annotations

import numpy as np
import pytest

from sfa.control.influence import compute_influence
from sfa.algorithms.sp import SignalPropagation

from _skip_helpers import no_cuda_reason

_NO_CUDA = no_cuda_reason()


def _W(N, seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, N)) * (0.5 / np.sqrt(N))
    np.fill_diagonal(W, 0.0)
    return W


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cpu_influence_dtype(dtype):
    W = _W(128, seed=11)
    S = compute_influence(
        W, alpha=0.5, beta=0.5, rtype="array",
        max_iter=2000, tol=1e-8, device="cpu", dtype=dtype)
    assert S.dtype == np.dtype(dtype)


def test_cpu_influence_fast_path_matches_iterative():
    """Closed-form LAPACK path must match the iterative result."""
    W = _W(256, seed=33).astype(np.float64)
    # Iterative (forced by get_iter=True)
    S_iter, _ = compute_influence(
        W, alpha=0.5, beta=0.5, rtype="array",
        max_iter=2000, tol=1e-12, device="cpu", get_iter=True)
    # Closed form (default when S is None and get_iter is False)
    S_solve = compute_influence(
        W, alpha=0.5, beta=0.5, rtype="array",
        max_iter=2000, tol=1e-12, device="cpu")
    np.testing.assert_allclose(S_solve, S_iter, atol=1e-9, rtol=1e-9)


def test_cpu_sp_preallocated_trajectory_shape():
    alg = SignalPropagation("SP")
    W = _W(64, seed=7)
    xi = np.zeros(64)
    rng = np.random.default_rng(7)
    b = rng.standard_normal(64) * 0.1
    x, trj = alg.propagate_iterative(
        W, xi, b, a=0.5, lim_iter=2000, tol=1e-8, get_trj=True)
    assert trj.ndim == 2
    assert trj.shape[1] == 64
    # First row is initial state, last row equals x.
    np.testing.assert_allclose(trj[0], xi)
    np.testing.assert_allclose(trj[-1], x)


@pytest.mark.skipif(_NO_CUDA is not None, reason=_NO_CUDA or "cuda")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cuda_sp_trajectory_matches_cpu(dtype):
    alg = SignalPropagation("SP")
    N = 128
    W = _W(N, seed=99)
    xi = np.zeros(N)
    rng = np.random.default_rng(99)
    b = rng.standard_normal(N) * 0.1

    x_cpu, trj_cpu = alg.propagate_iterative(
        W, xi, b, a=0.5, lim_iter=2000, tol=1e-7,
        get_trj=True, device="cpu", dtype=np.float64)
    x_gpu, trj_gpu = alg.propagate_iterative(
        W, xi, b, a=0.5, lim_iter=2000, tol=1e-7,
        get_trj=True, device="cuda:0", dtype=dtype)

    atol = 1e-4 if np.dtype(dtype) == np.float32 else 1e-6
    assert trj_gpu.ndim == 2 and trj_gpu.shape[1] == N
    # The CUDA path may terminate at a different iter count; we compare the
    # common prefix and the final state.
    common = min(trj_cpu.shape[0], trj_gpu.shape[0])
    np.testing.assert_allclose(
        trj_gpu[:common].astype(np.float64), trj_cpu[:common],
        atol=atol, rtol=atol)
    np.testing.assert_allclose(
        x_gpu.astype(np.float64), x_cpu, atol=atol, rtol=atol)


@pytest.mark.skipif(_NO_CUDA is not None, reason=_NO_CUDA or "cuda")
def test_cuda_influence_use_tf32_off_higher_precision():
    """Disabling TF32 must produce a tighter match to FP64 CPU baseline."""
    W = _W(256, seed=55).astype(np.float64)
    S_cpu = compute_influence(W, alpha=0.5, beta=0.5, rtype="array",
                              device="cpu", dtype=np.float64)
    S_tf32 = compute_influence(W, alpha=0.5, beta=0.5, rtype="array",
                               device="cuda:0", dtype=np.float32,
                               use_tf32=True)
    S_fp32 = compute_influence(W, alpha=0.5, beta=0.5, rtype="array",
                               device="cuda:0", dtype=np.float32,
                               use_tf32=False)
    err_tf32 = np.abs(S_tf32.astype(np.float64) - S_cpu).max()
    err_fp32 = np.abs(S_fp32.astype(np.float64) - S_cpu).max()
    # FP32 path should be at least as accurate as TF32 (often much more so).
    assert err_fp32 <= err_tf32 + 1e-12


def test_cpu_sp_nonconvergence_returns_last_iterate():
    """Non-converged runs must return the *last* iterate, not the one before.

    Regression test: the ping-pong swap optimization left the final swap in
    place when lim_iter was exhausted, so the function returned the
    second-to-last iterate (and a value that disagreed with trajectory[-1]).
    """
    alg = SignalPropagation("SP")
    N = 16
    W = _W(N, seed=3)
    xi = np.zeros(N)
    rng = np.random.default_rng(3)
    b = rng.standard_normal(N) * 0.1
    a = 0.5
    n_steps = 5  # small lim_iter + unreachable tol => guaranteed non-convergence

    x, num_iter = alg.propagate_iterative(
        W, xi, b, a=a, lim_iter=n_steps, tol=1e-30, get_trj=False)
    assert num_iter == n_steps

    # Manual reference: exactly n_steps fixed-point updates from xi.
    ref = xi.astype(np.float64).copy()
    for _ in range(n_steps):
        ref = a * W.dot(ref) + (1 - a) * b
    np.testing.assert_allclose(x, ref, rtol=1e-12, atol=1e-12)

    # The returned state must equal the last trajectory row.
    x_trj, trj = alg.propagate_iterative(
        W, xi, b, a=a, lim_iter=n_steps, tol=1e-30, get_trj=True)
    assert trj.shape[0] == n_steps + 1
    np.testing.assert_allclose(trj[-1], x_trj, rtol=1e-12, atol=1e-12)


def test_cpu_sp_zero_iterations_returns_initial():
    """lim_iter == 0 must return the initial state, not an uninitialized buffer."""
    alg = SignalPropagation("SP")
    N = 8
    W = _W(N, seed=5)
    xi = np.arange(N, dtype=np.float64)
    b = np.zeros(N)
    x, num_iter = alg.propagate_iterative(
        W, xi, b, a=0.5, lim_iter=0, tol=1e-5)
    assert num_iter == 0
    np.testing.assert_allclose(x, xi)
