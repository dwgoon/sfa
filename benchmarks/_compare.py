"""Compare native CUDA backend vs numpy CPU, with TF32 on and off."""
import time
import numpy as np
from sfa import _cuda
from sfa.control.influence import compute_influence

N = 512
rng = np.random.default_rng(42)
W = rng.standard_normal((N, N)).astype(np.float32) * 0.05
np.fill_diagonal(W, 0.0)

S_cpu, ni_cpu = compute_influence(
    W, alpha=0.5, beta=0.5, rtype="array",
    max_iter=2000, tol=1e-6, device="cpu", get_iter=True)
print(f"[cpu]            iters={ni_cpu:>4}  ||S||_F={np.linalg.norm(S_cpu):.6f}")

native = _cuda.native_module()
for use_tf32 in (False, True):
    t0 = time.perf_counter()
    S, ni = native.compute_influence_cuda(
        W, alpha=0.5, beta=0.5,
        max_iter=2000, tol=1e-6,
        device_id=0, check_every=8, use_tf32=use_tf32)
    t1 = time.perf_counter()
    diff = np.abs(S_cpu.astype(np.float64) - S.astype(np.float64)).max()
    tag = "TF32" if use_tf32 else "FP32"
    print(f"[cuda {tag}]  iters={ni:>4}  ||S||_F={np.linalg.norm(S):.6f}  "
          f"max|diff|={diff:.2e}  wall={1000*(t1-t0):6.1f} ms")
