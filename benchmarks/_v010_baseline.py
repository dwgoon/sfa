# -*- coding: utf-8 -*-
"""v0.1.0 reference implementations, lifted verbatim from commit 1b63dbe
(``Add MkDocs documentation for v0.1.0``) for benchmarking purposes.

The only edits are:
  * ``np.float`` -> ``np.float64`` (the alias was removed in NumPy 1.20).
  * Trailing comments stripped.

These functions are NOT exported through ``sfa``; they live here only as
a baseline against which to measure the 0.2.0.dev0 changes.
"""
import numpy as np


def compute_influence_cpu_v010(W, alpha=0.5, beta=0.5, S=None,
                               max_iter=1000, tol=1e-6, get_iter=False):
    """v0.1.0 CPU influence iteration (sfa/control/influence.py)."""
    N = W.shape[0]
    if S is not None:
        S1 = S
    else:
        S1 = np.eye(N, dtype=np.float64)

    I = np.eye(N, dtype=np.float64)
    S2 = np.zeros_like(W)
    aW = alpha * W
    for cnt in range(max_iter):
        S2[:, :] = S1.dot(aW) + I
        norm = np.linalg.norm(S2 - S1)
        if norm < tol:
            break
        S1[:, :] = S2

    S_fin = beta * S2
    if get_iter:
        return S_fin, cnt
    return S_fin


def propagate_iterative_v010(W, xi, b, a=0.5, lim_iter=1000, tol=1e-5,
                             get_trj=False):
    """v0.1.0 SignalPropagation.propagate_iterative (sfa/algorithms/sp.py)."""
    x0 = np.array(xi, dtype=np.float64)
    x_t1 = x0.copy()

    if get_trj:
        trj_x = []
        trj_x.append(x_t1.copy())

    num_iter = 0
    for i in range(lim_iter):
        x_t2 = a * W.dot(x_t1) + (1 - a) * b
        num_iter += 1
        if np.linalg.norm(x_t2 - x_t1) <= tol:
            break
        if get_trj:
            trj_x.append(x_t2)
        x_t1 = x_t2.copy()

    if get_trj is False:
        return x_t2, num_iter
    return x_t2, np.array(trj_x)
