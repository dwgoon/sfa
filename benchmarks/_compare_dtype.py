"""Verify the native CUDA backend at multiple dtypes vs numpy CPU.

Also smoke-test the new SignalPropagation CUDA path.
"""
import time
import numpy as np
from sfa import _cuda
from sfa.control.influence import compute_influence

native = _cuda.native_module()
print(f"device: {native.device_info(0)['name']}")
print()

# ---------- influence: dtype matrix ----------
N = 1024
rng = np.random.default_rng(42)
W = rng.standard_normal((N, N)).astype(np.float32) * (0.5 / np.sqrt(N))
np.fill_diagonal(W, 0.0)

S_cpu, ni_cpu = compute_influence(
    W, alpha=0.5, beta=0.5, rtype="array",
    max_iter=2000, tol=1e-6, device="cpu", get_iter=True)
print(f"[cpu f64]    iters={ni_cpu:>4}  ||S||_F={np.linalg.norm(S_cpu):.6f}")

for dtype in (np.float64, np.float32, np.float16):
    t0 = time.perf_counter()
    S, ni = compute_influence(
        W, alpha=0.5, beta=0.5, rtype="array",
        max_iter=2000, tol=1e-6, device="cuda:0",
        get_iter=True, dtype=dtype)
    t1 = time.perf_counter()
    diff = np.abs(S_cpu.astype(np.float64) - S.astype(np.float64)).max()
    print(f"[cuda {np.dtype(dtype).name:>6}]  iters={ni:>4}  "
          f"||S||_F={np.linalg.norm(S):.6f}  max|diff|={diff:.2e}  "
          f"dtype_out={S.dtype}  wall={1000*(t1-t0):6.1f} ms")

print()
print("---- SignalPropagation (CUDA) ----")
from sfa.algorithms.sp import SignalPropagation
alg = SignalPropagation("SP")
W64 = rng.standard_normal((N, N)).astype(np.float64) * (0.5 / np.sqrt(N))
np.fill_diagonal(W64, 0.0)
xi = np.zeros(N)
b  = rng.standard_normal(N) * 0.1
a  = 0.5

x_cpu, ni_cpu = alg.propagate_iterative(
    W64, xi, b, a=a, lim_iter=2000, tol=1e-6, get_trj=False, device="cpu")
print(f"[cpu  f64]   iters={ni_cpu:>4}  ||x||={np.linalg.norm(x_cpu):.6f}")

for dtype in (np.float64, np.float32, np.float16):
    t0 = time.perf_counter()
    x, ni = alg.propagate_iterative(
        W64, xi, b, a=a, lim_iter=2000, tol=1e-6,
        get_trj=False, device="cuda:0", dtype=dtype)
    t1 = time.perf_counter()
    diff = np.abs(x_cpu - x.astype(np.float64)).max()
    print(f"[cuda {np.dtype(dtype).name:>6}] iters={ni:>4}  "
          f"||x||={np.linalg.norm(x.astype(np.float64)):.6f}  max|diff|={diff:.2e}  "
          f"dtype_out={x.dtype}  wall={1000*(t1-t0):6.1f} ms")
