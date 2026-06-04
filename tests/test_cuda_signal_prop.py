"""Correctness tests for the native CUDA SignalPropagation backend."""
from __future__ import annotations

import numpy as np
import pytest

from _skip_helpers import no_cuda_reason

_reason = no_cuda_reason()
pytestmark = pytest.mark.skipif(_reason is not None, reason=_reason or "cuda")


def _setup(N, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, N)).astype(np.float64) * (0.5 / np.sqrt(N))
    np.fill_diagonal(W, 0.0)
    xi = np.zeros(N)
    b = rng.standard_normal(N) * 0.1
    return W, xi, b


@pytest.mark.parametrize("N", [64, 256])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cuda_signal_prop_matches_cpu(N, dtype):
    from sfa.algorithms.sp import SignalPropagation
    alg = SignalPropagation("SP")
    W, xi, b = _setup(N, seed=N)
    a, tol = 0.5, 1e-6
    x_cpu, _ = alg.propagate_iterative(
        W, xi, b, a=a, lim_iter=2000, tol=tol, device="cpu")
    x_gpu, ni = alg.propagate_iterative(
        W, xi, b, a=a, lim_iter=2000, tol=tol,
        device="cuda:0", dtype=dtype)
    # f32 path with TF32 has ~1e-5 absolute noise floor; f64 is near-exact.
    atol = 1e-4 if np.dtype(dtype) == np.float32 else 1e-6
    np.testing.assert_allclose(
        x_gpu.astype(np.float64), x_cpu, atol=atol, rtol=atol,
        err_msg=f"CUDA signal-prop mismatch N={N} dtype={dtype}")
    assert 1 <= ni < 2000


def test_cuda_signal_prop_float16_runs():
    """float16 has too little dynamic range to match f64 tightly; we only
    check that the CUDA path produces a finite result and a reasonable
    iteration count."""
    from sfa.algorithms.sp import SignalPropagation
    alg = SignalPropagation("SP")
    W, xi, b = _setup(128, seed=128)
    x, ni = alg.propagate_iterative(
        W, xi, b, a=0.5, lim_iter=2000, tol=1e-4,
        device="cuda:0", dtype=np.float16)
    assert np.isfinite(x.astype(np.float64)).all()
    assert 1 <= ni < 2000
