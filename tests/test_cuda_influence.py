# -*- coding: utf-8 -*-
"""Correctness tests for the native CUDA influence backend (sfa 0.2.x).

These are skipped automatically if the native extension was not built
(e.g. on a machine without a CUDA toolchain).
"""
from __future__ import annotations

import numpy as np
import pytest

from sfa.control.influence import compute_influence

from _skip_helpers import no_cuda_reason

_reason = no_cuda_reason()
pytestmark = pytest.mark.skipif(_reason is not None, reason=_reason or "cuda")


def _random_weight_matrix(n, seed=0):
    rng = np.random.default_rng(seed)
    # Sparse-ish, spectral radius < 1/alpha to ensure convergence.
    W = rng.standard_normal((n, n)).astype(np.float32) * 0.05
    # zero out the diagonal (a common convention for signal flow)
    np.fill_diagonal(W, 0.0)
    return W


@pytest.mark.parametrize("n", [64, 256, 512])
def test_native_matches_cpu(n):
    W = _random_weight_matrix(n, seed=42)
    alpha, beta = 0.5, 0.5
    max_iter, tol = 2000, 1e-7

    S_cpu = compute_influence(
        W, alpha=alpha, beta=beta, rtype="array",
        max_iter=max_iter, tol=tol, device="cpu")

    S_gpu, _ = compute_influence(
        W, alpha=alpha, beta=beta, rtype="array",
        max_iter=max_iter, tol=tol, device="cuda:0",
        get_iter=True)

    # TF32 in cuBLAS reduces effective mantissa to 10 bits, so we allow a
    # generous absolute + relative tolerance.
    np.testing.assert_allclose(
        S_gpu.astype(np.float64),
        S_cpu.astype(np.float64),
        atol=1e-3, rtol=1e-3,
        err_msg=f"CUDA influence mismatch for N={n}")


def test_native_iter_count_reasonable():
    W = _random_weight_matrix(256, seed=1)
    _, num_iter = compute_influence(
        W, alpha=0.5, beta=0.5, rtype="array",
        max_iter=2000, tol=1e-6, device="cuda:0", get_iter=True)
    # Should converge well before the cap on a well-conditioned random W.
    assert 1 <= num_iter < 2000
