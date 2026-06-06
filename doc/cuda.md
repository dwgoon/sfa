# CUDA backend

SFA's CUDA backend is an optional native extension (`sfa._cuda._native`,
built from C++/CUDA sources under `sfa/_cuda/src`) that accelerates the
two main iterative loops:

- the **influence matrix** iteration $S(t+1) = S(t)(\alpha W) + I$, used
  by `sfa.control.compute_influence` when `device='cuda:N'`;
- the **signal propagation** iteration $x(t+1) = \alpha W x(t) + (1 -
  \alpha) b$, used by `SignalPropagation.propagate_iterative` when
  `device='cuda:N'`.

The extension is loaded lazily. When it is not present (CPU-only wheel,
build skipped, or runtime DLL missing), `sfa._cuda.HAS_NATIVE` is
`False` and `device='cuda:N'` transparently falls back to the CuPy path
or, ultimately, to the NumPy CPU path. Code that uses `device='cuda:0'`
keeps working in any environment.

## Architecture

Per iteration the backend issues:

1. **cuBLAS GEMM** (`Sgemm` / `Dgemm` / `GemmEx` for FP16) for the dense
   matrix product. The compute mode is selected by `dtype`:
    - `float32` + `use_tf32=True` (default): TF32 Tensor Cores.
    - `float32` + `use_tf32=False`: plain FP32.
    - `float64`: standard FP64 DGEMM. Slow on consumer Ada but exact.
    - `float16`: HMMA with `CUBLAS_COMPUTE_16F` (FP16 accumulate).
2. **One hand-written fused kernel** that combines the algorithm-
   specific addition (`+ I` for influence; `+ (1-alpha)*b` for signal
   propagation) with a Frobenius / $L_2$ diff-squared reduction in a
   single pass over the output, ending with a typed `atomicAdd` into a
   device-resident scalar.
3. **CUDA Graph capture/replay** over `check_every` iterations to
   amortize per-launch overhead. The host periodically copies the
   scalar back, checks the tolerance, and breaks out when either
   `||S(t+1) - S(t)||_F^2 <= tol^2` or the diff stops decreasing (the
   plateau detector handles TF32/FP16 noise floors).

## dtype contract

The user-selected dtype propagates through every layer:

| Layer                              | Effect of `dtype=T`                                          |
|------------------------------------|--------------------------------------------------------------|
| Python (`compute_influence(...)`) | Casts `W` (and `xi`, `b`) to `T` before calling C++.         |
| pybind11 dispatch                 | Routes to `run_with_dtype<T>` instantiation.                 |
| Device buffers                    | Allocated as `sizeof(T)` per element.                        |
| cuBLAS GEMM / GEMV                | `Sgemm` / `Dgemm` / `GemmEx<COMPUTE_16F>` selected by `T`.   |
| Fused kernel arithmetic           | All adds / mults in `T` (cuda_fp16.h operator overloads).    |
| Convergence accumulator           | `atomicAdd<T>` into a single-`T` device scalar.              |
| Output dtype                      | Matches `T`. `dtype=float16` produces a `float16` `ndarray`. |

`alpha`, `beta`, and `tol` are forwarded as C++ `double` so the FP64
path keeps full precision; the FP32 and FP16 paths cast at the cuBLAS
call site.

Strict-dtype consequences for `float16`:

- HMMA accumulate is in FP16, not FP32. Long inner products in matrices
  with large magnitudes can overflow. Use `float32` instead when the
  matrix entries push close to the FP16 normal range.
- The Frobenius diff accumulator is FP16. Differences below ~6e-5
  underflow to zero, so very tight `tol` does not necessarily mean
  "very tight result"; the plateau detector ends the loop at the FP16
  noise floor regardless.

## Tuning knobs

`compute_influence` (and `SignalPropagation.propagate_iterative`)
forward the following options unchanged to the CUDA backend:

| Parameter      | Default       | Effect                                                          |
|----------------|---------------|-----------------------------------------------------------------|
| `dtype`        | `None`        | One of `float32`, `float64`, `float16`. `None` infers from `W.dtype`. |
| `check_every`  | `8`           | Number of iterations between host syncs (and the size of the captured CUDA Graph batch). `1` disables graph capture. |
| `use_tf32`     | `True`        | Enable `CUBLAS_TF32_TENSOR_OP_MATH` for the FP32 path. No effect on FP64 / FP16. |

## Trajectory mode (signal propagation)

When `propagate_iterative(..., get_trj=True, device='cuda:N')` is
called, the backend allocates a `(max_iter + 1) * N` device-resident
buffer, writes one row per iteration with a `DeviceToDevice` memcpy,
and downloads the actual `(num_iter + 1, N)` slice with a single
`DeviceToHost` transfer at the end. CUDA Graph capture is disabled in
trajectory mode because each iteration writes to a different row
offset.

The CPU trajectory path mirrors this with a single pre-allocated
`(lim_iter + 1, N)` NumPy buffer (no Python-list-of-arrays growth).

## Backend dispatch and fallback chain

```
device='cuda:N'
   |
   | native available and a GPU is visible?
   |     yes -> sfa._cuda._native (this page)
   |     no  -> try cupy (legacy GPU backend)
   |               yes -> sfa.control.influence._compute_influence_gpu
   |               no  -> sfa.control.influence._compute_influence_cpu

device='gpu:N'    -> cupy path (legacy; kept for backward compatibility)
device='cpu'      -> NumPy + scipy.linalg.solve (LAPACK closed form)
```

The fallback is silent. Use `sfa._cuda.HAS_NATIVE` and
`sfa._cuda.native_module().device_count()` to introspect explicitly.

## Performance

Indicative numbers on a single RTX 4090 (24 GB, sm_89) versus the CPU
LAPACK closed-form path on a typical desktop:

| N      | CPU `solve` (s, f64) | CUDA `float32` (s) | speedup |
|--------|----------------------|--------------------|---------|
| 512    | 0.151                | 0.0024             | 63x     |
| 1024   | 0.369                | 0.0053             | 70x     |
| 2048   | 0.528                | 0.027              | 19x     |
| 4096   | 1.693                | 0.165              | 10x     |

Numbers are best-of-3 wallclock from `benchmarks/bench_influence.py`; the
CPU column reports the LAPACK closed-form path, not the iterative one
(which is slower).

## Building from source for a specific arch

The shipped wheels carry SASS for SM 7.0..9.0. To target a single arch
locally (faster build):

```bash
SFA_CUDA_ARCH=sm_89 pip install -e .
```

To produce a CUDA wheel with a custom arch list, see the
[Install](install.md) page.
