"""Native CUDA influence backend across dtypes."""
from __future__ import annotations

import numpy as np
import pytest

from sfa.control.influence import compute_influence

from _skip_helpers import no_cuda_reason

_reason = no_cuda_reason()
pytestmark = pytest.mark.skipif(_reason is not None, reason=_reason or "cuda")


def _W(N, seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, N)).astype(np.float64) * (0.5 / np.sqrt(N))
    np.fill_diagonal(W, 0.0)
    return W


@pytest.mark.parametrize("N", [128, 512])
@pytest.mark.parametrize("dtype,atol", [
    (np.float64, 1e-6),
    (np.float32, 1e-3),
    # float16 has only ~3 decimal digits; we accept a coarse match.
    (np.float16, 5e-2),
])
def test_cuda_influence_dtype_matches_cpu(N, dtype, atol):
    W = _W(N, seed=N)
    alpha, beta = 0.5, 0.5
    S_cpu = compute_influence(
        W, alpha=alpha, beta=beta, rtype="array",
        max_iter=2000, tol=1e-7, device="cpu")
    S_gpu = compute_influence(
        W, alpha=alpha, beta=beta, rtype="array",
        max_iter=2000, tol=1e-6,
        device="cuda:0", dtype=dtype)
    assert S_gpu.dtype == np.dtype(dtype)
    np.testing.assert_allclose(
        S_gpu.astype(np.float64), S_cpu, atol=atol, rtol=atol,
        err_msg=f"CUDA influence mismatch N={N} dtype={dtype}")


def test_cuda_influence_dtype_inferred_from_W():
    W = _W(128, seed=7).astype(np.float32)
    S = compute_influence(
        W, alpha=0.5, beta=0.5, rtype="array",
        max_iter=2000, tol=1e-6, device="cuda:0")
    assert S.dtype == np.float32
