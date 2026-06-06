from sfa._cuda import native_module, HAS_NATIVE
print("HAS_NATIVE:", HAS_NATIVE)
m = native_module()
print("module:", m)
print("device_count:", m.device_count())
print("device_info(0):", m.device_info(0))

import numpy as np
N = 256
rng = np.random.default_rng(42)
W = (rng.standard_normal((N, N)).astype(np.float32) * 0.05)
np.fill_diagonal(W, 0.0)

S, ni = m.compute_influence_cuda(W, alpha=0.5, beta=0.5,
                                  max_iter=2000, tol=1e-6,
                                  device_id=0, check_every=8,
                                  use_tf32=True)
print("shape:", S.shape, "dtype:", S.dtype, "iters:", ni)
print("S[:3,:3]=", S[:3, :3])
