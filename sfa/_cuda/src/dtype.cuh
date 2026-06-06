// Dtype traits + device/host conversions for sfa's CUDA kernels.
//
// Supported scalar types:
//   float                  (FP32; with optional TF32 Tensor-Core math mode)
//   double                 (FP64; slow on consumer Ada but exact)
//   __half                 (FP16; cuBLAS GemmEx with FP16 accumulate)
//
// "Strict dtype" contract (v0.2.0.dev0):
//   The user-selected dtype determines storage, GEMM compute type, fused-
//   kernel arithmetic, and the device-side convergence accumulator. The
//   library does not silently promote to a wider type. Host-side control
//   flow (plateau detection, tolerance comparison) reads back the device
//   accumulator as `double` only to keep the iteration scheduler
//   numerically robust; the underlying computation stays in T.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace sfa {

// Compile-time tag identifying the storage dtype.
enum class DType { F32, F64, F16 };

// ----- device-side conversion helpers ----------------------------------------
template <class T> __device__ inline float to_float(T x);
template <> __device__ inline float to_float<float>(float x)   { return x; }
template <> __device__ inline float to_float<double>(double x) { return (float)x; }
template <> __device__ inline float to_float<__half>(__half x) { return __half2float(x); }

template <class T> __device__ inline double to_double(T x);
template <> __device__ inline double to_double<float>(float x)   { return (double)x; }
template <> __device__ inline double to_double<double>(double x) { return x; }
template <> __device__ inline double to_double<__half>(__half x) { return (double)__half2float(x); }

template <class T> __device__ inline T from_float(float x);
template <> __device__ inline float  from_float<float>(float x)  { return x; }
template <> __device__ inline double from_float<double>(float x) { return (double)x; }
template <> __device__ inline __half from_float<__half>(float x) { return __float2half(x); }

template <class T> __device__ inline T from_double(double x);
template <> __device__ inline float  from_double<float>(double x)  { return (float)x; }
template <> __device__ inline double from_double<double>(double x) { return x; }
template <> __device__ inline __half from_double<__half>(double x){ return __float2half((float)x); }

// Strict accumulator policy: Acc == T for all storage types. The Frobenius /
// L2 diff reduction stays in the user-chosen dtype on the device.
template <class T> struct AccTraits { using Acc = T; };

// ----- host-side conversion (called only from the iteration driver) ---------
inline double host_to_double(float x)   { return (double)x; }
inline double host_to_double(double x)  { return x; }
inline double host_to_double(__half x)  { return (double)__half2float(x); }

// ----- device-side typed zero / one literals (avoids brace-init pitfalls) ----
template <class T> __device__ inline T t_zero();
template <> __device__ inline float  t_zero<float>()  { return 0.0f; }
template <> __device__ inline double t_zero<double>() { return 0.0; }
template <> __device__ inline __half t_zero<__half>() { return __float2half(0.0f); }

template <class T> __device__ inline T t_one();
template <> __device__ inline float  t_one<float>()  { return 1.0f; }
template <> __device__ inline double t_one<double>() { return 1.0; }
template <> __device__ inline __half t_one<__half>() { return __float2half(1.0f); }

} // namespace sfa
