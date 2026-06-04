// Influence-matrix fixed-point iteration on NVIDIA Ada (sm_89).
//
// Math:  S(t+1) = S(t) * (alpha * W) + I,  stop when ||S(t+1)-S(t)||_F <= tol.
//
// The kernels here are templated on the storage scalar type T:
//   T = float  : cuBLAS SGEMM, optionally TF32 Tensor-Core math mode.
//   T = double : cuBLAS DGEMM. Functional but slow on consumer Ada (FP64 = 1/64
//                FP32 throughput on RTX 4090).
//   T = __half : cuBLAS GemmEx with FP16 storage + FP32 accumulate (HMMA).
//
// All matrices on the device follow the same row-major-on-storage convention
// (see bindings.cpp): bytes are row-major, cuBLAS sees them as column-major
// which is the mathematical transpose, and we issue the GEMM with operands
// swapped to land back on the correct row-major bytes for the output.

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <type_traits>
#include <utility>

#include "common.cuh"
#include "dtype.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#ifndef SFA_TILE
#define SFA_TILE 16
#endif

namespace sfa {

// =============================================================================
// Fused kernel: S_next = P + I_diag ; reduce sum((S_next - S_in)^2).
//
// Strict-dtype version: the device accumulator `d_diff_sq` is of the same
// scalar type T as the storage matrices. The caller must zero it before
// each launch (cudaMemsetAsync over sizeof(T) bytes).
// =============================================================================
// Strict-dtype fused kernel:
//   S_out[i,j] = P[i,j] + I_diag(i,j)
//   d_diff_sq += sum_ij (S_out[i,j] - S_in[i,j])^2
// All arithmetic is performed in T (storage type). For T=__half this relies
// on the operator+/-/* overloads in cuda_fp16.h and the sm_70+ overload of
// atomicAdd(__half*, __half).
template <class T>
__global__ void fused_add_identity_and_norm(
        const T* __restrict__ P,
        const T* __restrict__ S_in,
        T*       __restrict__ S_out,
        T*       __restrict__ d_diff_sq,
        int N)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    T local = t_zero<T>();

    if (col < N && row < N) {
        const int idx = row * N + col;
        T p     = P[idx];
        T ident = (row == col) ? t_one<T>() : t_zero<T>();
        T val   = p + ident;
        T sin   = S_in[idx];
        T d     = val - sin;
        local   = d * d;
        S_out[idx] = val;
    }

    __shared__ T sdata[SFA_TILE * SFA_TILE];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    sdata[tid] = local;
    __syncthreads();

    // Tree reduction (block is 256 = power of two).
    for (int s = (blockDim.x * blockDim.y) >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(d_diff_sq, sdata[0]);
    }
}

// =============================================================================
// scale_kernel: S <- beta * S (element-wise), in place.
//
// `beta` arrives as double to preserve the caller's precision; the FP64
// path multiplies in double, FP32 / FP16 paths cast down inside the kernel.
// =============================================================================
template <class T>
__global__ void scale_kernel(T* S, double beta, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if constexpr (std::is_same_v<T, double>) {
            S[i] = S[i] * beta;
        } else {
            float v = to_float<T>(S[i]);
            S[i] = from_float<T>(v * (float)beta);
        }
    }
}

// =============================================================================
// GEMM dispatch per dtype.
//
// alpha and beta_blas arrive as `double` to preserve full precision from
// Python (the FP64 path uses them as-is; FP32 and FP16 paths cast down at
// the call site).
// =============================================================================
template <class T>
inline void gemm_NN(cublasHandle_t handle,
                    int N, double alpha, double beta_blas,
                    const T* A, const T* B, T* C);

template <>
inline void gemm_NN<float>(cublasHandle_t handle,
                           int N, double alpha, double beta_blas,
                           const float* A, const float* B, float* C)
{
    const float a = (float)alpha;
    const float b = (float)beta_blas;
    SFA_CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &a, A, N, B, N,
        &b, C, N));
}

template <>
inline void gemm_NN<double>(cublasHandle_t handle,
                            int N, double alpha, double beta_blas,
                            const double* A, const double* B, double* C)
{
    SFA_CUBLAS_CHECK(cublasDgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha, A, N, B, N,
        &beta_blas, C, N));
}

template <>
inline void gemm_NN<__half>(cublasHandle_t handle,
                            int N, double alpha, double beta_blas,
                            const __half* A, const __half* B, __half* C)
{
    // Strict FP16 path: CUBLAS_COMPUTE_16F => Tensor Core HMMA with FP16
    // accumulate. Scalars are __half, matching the compute type. This is
    // the user's contract for dtype=float16 even though FP32 accumulate
    // would normally be more numerically stable.
    const __half a = __float2half((float)alpha);
    const __half b = __float2half((float)beta_blas);
    SFA_CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &a,
        A, CUDA_R_16F, N,
        B, CUDA_R_16F, N,
        &b,
        C, CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// =============================================================================
// Templated iteration driver.
// =============================================================================
template <class T>
int run_influence_iteration_impl(
        cublasHandle_t handle,
        cudaStream_t   stream,
        int            N,
        double         alpha,
        const T*       d_W,
        T*             d_S0,
        T*             d_scratch,  // size 2 * N*N elements of T
        T*             d_diff,     // single T element (strict-dtype accumulator)
        int            max_iter,
        double         tol,
        int            check_every)
{
    T* d_P       = d_scratch;
    T* d_S_other = d_scratch + (size_t)N * N;

    T* d_S_cur = d_S0;
    T* d_S_nxt = d_S_other;

    dim3 block(SFA_TILE, SFA_TILE);
    dim3 grid((N + SFA_TILE - 1) / SFA_TILE,
              (N + SFA_TILE - 1) / SFA_TILE);

    // tol_sq is also recomputed inside the inner loop as `tol_sq_d` (double);
    // we keep this no longer-used variable removed.

    bool use_graph = (check_every > 1);
    if (use_graph && (check_every % 2 == 1)) {
        check_every += 1;
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    auto one_step = [&](T* S_cur_ptr, T* S_nxt_ptr) {
        // P = (alpha * W) * S_cur in row-major; see comment in the previous
        // implementation for the cuBLAS-column-major <-> row-major argument.
        gemm_NN<T>(handle, N, alpha, 0.0, d_W, S_cur_ptr, d_P);
        SFA_CUDA_CHECK(cudaMemsetAsync(d_diff, 0, sizeof(T), stream));
        fused_add_identity_and_norm<T><<<grid, block, 0, stream>>>(
            d_P, S_cur_ptr, S_nxt_ptr, d_diff, N);
    };

    if (use_graph) {
        // Warm-up cuBLAS so its lazy init does not get recorded into the graph.
        gemm_NN<T>(handle, N, 0.0, 0.0, d_W, d_S_cur, d_P);
        SFA_CUDA_CHECK(cudaStreamSynchronize(stream));

        SFA_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
        one_step(d_S_cur, d_S_nxt);
        one_step(d_S_nxt, d_S_cur);
        SFA_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        SFA_CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    }

    // Device-side accumulator h_diff is read back as type T then converted
    // to double for the host-side scheduler. The actual numeric value of
    // d_diff is whatever T's arithmetic produced (e.g. fp16 may underflow
    // small differences to zero); converting it to double on the host
    // doesn't recover precision but lets the plateau / tol check use a
    // single uniform code path.
    T h_diff_T;
    double h_diff = 0.0;
    double h_prev = 1e308;
    int plateau = 0;
    const int captured_steps = use_graph ? 2 : 1;
    const double tol_sq_d = (double)tol * (double)tol;

    int iter = 0;
    while (iter < max_iter) {
        if (use_graph) {
            int batch = check_every / captured_steps;
            for (int k = 0; k < batch && iter < max_iter; ++k) {
                SFA_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
                iter += captured_steps;
            }
        } else {
            one_step(d_S_cur, d_S_nxt);
            std::swap(d_S_cur, d_S_nxt);
            iter++;
        }

        SFA_CUDA_CHECK(cudaMemcpyAsync(&h_diff_T, d_diff, sizeof(T),
                                       cudaMemcpyDeviceToHost, stream));
        SFA_CUDA_CHECK(cudaStreamSynchronize(stream));
        h_diff = host_to_double(h_diff_T);

        if (getenv("SFA_CUDA_DEBUG")) {
            fprintf(stderr,
                "[sfa-cuda] iter=%5d  diff_sq=%.6e  tol_sq=%.6e  plateau=%d\n",
                iter, h_diff, tol_sq_d, plateau);
        }

        if (h_diff <= tol_sq_d) break;

        if (h_diff > 0.99 * h_prev) {
            if (++plateau >= 2) break;
        } else {
            plateau = 0;
        }
        h_prev = h_diff;
    }

    // Make sure the final result lives in d_S0.
    if (d_S_cur != d_S0) {
        SFA_CUDA_CHECK(cudaMemcpyAsync(
            d_S0, d_S_cur, sizeof(T) * (size_t)N * N,
            cudaMemcpyDeviceToDevice, stream));
        SFA_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph)      cudaGraphDestroy(graph);

    return iter;
}

template <class T>
void scale_inplace_impl(cudaStream_t stream, T* d_S, double beta, size_t n)
{
    const int tpb = 256;
    int blocks = (int)((n + tpb - 1) / tpb);
    scale_kernel<T><<<blocks, tpb, 0, stream>>>(d_S, beta, n);
}

// =============================================================================
// Explicit instantiations exposed to bindings.cpp.
// =============================================================================
template int run_influence_iteration_impl<float>(
    cublasHandle_t, cudaStream_t, int, double,
    const float*, float*, float*, float*,
    int, double, int);

template int run_influence_iteration_impl<double>(
    cublasHandle_t, cudaStream_t, int, double,
    const double*, double*, double*, double*,
    int, double, int);

template int run_influence_iteration_impl<__half>(
    cublasHandle_t, cudaStream_t, int, double,
    const __half*, __half*, __half*, __half*,
    int, double, int);

template void scale_inplace_impl<float >(cudaStream_t, float*,  double, size_t);
template void scale_inplace_impl<double>(cudaStream_t, double*, double, size_t);
template void scale_inplace_impl<__half>(cudaStream_t, __half*, double, size_t);

}  // namespace sfa
