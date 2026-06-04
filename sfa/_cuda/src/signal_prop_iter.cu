// Signal-propagation fixed-point iteration on NVIDIA Ada (sm_89).
//
// Math:  x(t+1) = alpha * W * x(t) + (1 - alpha) * b,
//        stop when ||x(t+1) - x(t)||_2 <= tol.
//
// Per iteration: 1 GEMV (alpha * W * x), 1 axpy/add of (1-alpha)*b, and a
// fused L2-diff reduction. We use cuBLAS for the GEMV (S/D/GemvEx) and a
// small custom kernel for the add-bias + diff-norm.
//
// We also CUDA-Graph the iteration (pair of steps, ping-pong) like the
// influence kernel, so the inner loop has zero launch overhead between
// host syncs.

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

namespace sfa {

// =============================================================================
// Fused vector kernel:
//     x_out[i] = p[i] + (1 - alpha) * b[i]
//     local    = (x_out[i] - x_in[i])^2 ;  block-reduce -> atomicAdd into d_diff
// =============================================================================
// Strict-dtype fused kernel:
//   x_out[i] = p[i] + (1 - alpha) * b[i]
//   d_diff  += sum_i (x_out[i] - x_in[i])^2
// Arithmetic is in T (storage type). The one_minus_alpha bias scale is
// passed in as double and converted to T inside the kernel for the FP64
// path; the FP32/FP16 paths use the T-side conversion. atomicAdd to T is
// supported on sm_70+ for __half (cuda_fp16.h).
template <class T>
__global__ void fused_add_bias_and_norm(
        const T*  __restrict__ p,        // p = alpha * W * x_in (pre-scaled)
        const T*  __restrict__ b,
        const T*  __restrict__ x_in,
        T*        __restrict__ x_out,
        T*        __restrict__ d_diff,
        double    one_minus_alpha,
        int N)
{
    extern __shared__ unsigned char smem_raw[];
    T* sdata = reinterpret_cast<T*>(smem_raw);

    const int tid = threadIdx.x;
    T local = t_zero<T>();

    int i = blockIdx.x * blockDim.x + tid;
    if (i < N) {
        T p_i  = p[i];
        T b_i  = b[i];
        T xin  = x_in[i];
        T oma  = from_double<T>(one_minus_alpha);
        T val  = p_i + oma * b_i;
        T d    = val - xin;
        local  = d * d;
        x_out[i] = val;
    }

    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_diff, sdata[0]);
}

// =============================================================================
// GEMV dispatch per dtype.
//
// alpha and beta_s arrive as `double` so the FP64 path uses full precision;
// FP32 and FP16 paths cast down at the call site.
// =============================================================================
template <class T>
inline void gemv(cublasHandle_t handle, int N, double alpha,
                 const T* A, const T* x, double beta_s, T* y);

template <>
inline void gemv<float>(cublasHandle_t handle, int N, double alpha,
                        const float* A, const float* x,
                        double beta_s, float* y)
{
    const float a = (float)alpha, b = (float)beta_s;
    SFA_CUBLAS_CHECK(cublasSgemv(
        handle, CUBLAS_OP_T,
        N, N,
        &a, A, N,
        x, 1,
        &b, y, 1));
}

template <>
inline void gemv<double>(cublasHandle_t handle, int N, double alpha,
                         const double* A, const double* x,
                         double beta_s, double* y)
{
    SFA_CUBLAS_CHECK(cublasDgemv(
        handle, CUBLAS_OP_T,
        N, N,
        &alpha, A, N,
        x, 1,
        &beta_s, y, 1));
}

// cuBLAS lacks a GemvEx for FP16; cast to GemmEx with a 1-column matrix.
// Strict FP16: CUBLAS_COMPUTE_16F => Tensor Core HMMA with FP16 accumulate
// and __half scalars (matches dtype contract).
template <>
inline void gemv<__half>(cublasHandle_t handle, int N, double alpha,
                         const __half* A, const __half* x,
                         double beta_s, __half* y)
{
    const __half a = __float2half((float)alpha);
    const __half b = __float2half((float)beta_s);
    SFA_CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        /*m=*/N, /*n=*/1, /*k=*/N,
        &a,
        A, CUDA_R_16F, /*lda=*/N,
        x, CUDA_R_16F, /*ldb=*/N,
        &b,
        y, CUDA_R_16F, /*ldc=*/N,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// =============================================================================
// Templated driver.
//
//   d_x_io     in: x(0); out: x_converged (we always copy back into d_x_io).
//   d_x_other  scratch ping-pong vector of size N.
// =============================================================================
template <class T>
int run_signal_prop_impl(
        cublasHandle_t handle,
        cudaStream_t   stream,
        int            N,
        double         alpha,
        const T*       d_W,
        const T*       d_b,
        T*             d_x_io,
        T*             d_x_other,
        T*             d_diff,     // single T (strict-dtype accumulator)
        int            max_iter,
        double         tol,
        int            check_every,
        T*             d_trj /* nullable; shape (max_iter+1) * N */)
{
    const bool want_trj = (d_trj != nullptr);

    // Separate buffer for the GEMV target: gemv writes p = alpha*W*x_cur,
    // the fused kernel then consumes p, b, x_cur and writes x_nxt.
    T* d_p = nullptr;
    SFA_CUDA_CHECK(cudaMalloc(&d_p, sizeof(T) * (size_t)N));

    T* d_x_cur = d_x_io;
    T* d_x_nxt = d_x_other;
    // The fused kernel receives `one_minus_alpha` as double and converts
    // to T inside; FP64 path keeps full precision, FP32/FP16 rounds.
    const double one_minus_alpha = 1.0 - alpha;

    const int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    size_t shmem = (size_t)tpb * sizeof(T);

    auto one_step = [&](T* x_cur, T* x_nxt) {
        // p = alpha * W * x_cur
        gemv<T>(handle, N, alpha, d_W, x_cur, 0.0, d_p);
        SFA_CUDA_CHECK(cudaMemsetAsync(d_diff, 0, sizeof(T), stream));
        fused_add_bias_and_norm<T><<<blocks, tpb, shmem, stream>>>(
            d_p, d_b, x_cur, x_nxt, d_diff, one_minus_alpha, N);
    };

    const double tol_sq_d = tol * tol;

    // CUDA Graph captures fixed pointers; the trajectory row target moves
    // every iter, so disable graph mode when the caller wants the path.
    bool use_graph = (check_every > 1) && !want_trj;
    if (use_graph && (check_every % 2 == 1)) check_every += 1;

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t gexec = nullptr;

    if (use_graph) {
        // Warm up cuBLAS lazy init outside the capture.
        gemv<T>(handle, N, 0.0, d_W, d_x_cur, 0.0, d_p);
        SFA_CUDA_CHECK(cudaStreamSynchronize(stream));

        SFA_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
        one_step(d_x_cur, d_x_nxt);
        one_step(d_x_nxt, d_x_cur);
        SFA_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        SFA_CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
    }

    if (want_trj) {
        // trj[0] = initial x  (d_x_io == d_x_cur at this point).
        SFA_CUDA_CHECK(cudaMemcpyAsync(
            d_trj, d_x_io, sizeof(T) * (size_t)N,
            cudaMemcpyDeviceToDevice, stream));
    }

    T h_diff_T;
    double h_diff = 0.0;
    double h_prev = 1e308;
    int plateau = 0;
    const int captured_steps = use_graph ? 2 : 1;

    int iter = 0;
    while (iter < max_iter) {
        if (use_graph) {
            int batch = check_every / captured_steps;
            for (int k = 0; k < batch && iter < max_iter; ++k) {
                SFA_CUDA_CHECK(cudaGraphLaunch(gexec, stream));
                iter += captured_steps;
            }
        } else {
            one_step(d_x_cur, d_x_nxt);
            std::swap(d_x_cur, d_x_nxt);
            iter++;
            // Record this iteration's state into the trajectory buffer
            // (row index = iter; we already wrote row 0 above).
            if (want_trj) {
                SFA_CUDA_CHECK(cudaMemcpyAsync(
                    d_trj + (size_t)iter * N, d_x_cur, sizeof(T) * (size_t)N,
                    cudaMemcpyDeviceToDevice, stream));
            }
        }

        SFA_CUDA_CHECK(cudaMemcpyAsync(&h_diff_T, d_diff, sizeof(T),
                                       cudaMemcpyDeviceToHost, stream));
        SFA_CUDA_CHECK(cudaStreamSynchronize(stream));
        h_diff = host_to_double(h_diff_T);

        if (getenv("SFA_CUDA_DEBUG")) {
            fprintf(stderr,
                "[sfa-sp]  iter=%5d  diff_sq=%.6e  tol_sq=%.6e  plateau=%d\n",
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

    if (d_x_cur != d_x_io) {
        SFA_CUDA_CHECK(cudaMemcpyAsync(
            d_x_io, d_x_cur, sizeof(T) * (size_t)N,
            cudaMemcpyDeviceToDevice, stream));
        SFA_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (gexec) cudaGraphExecDestroy(gexec);
    if (graph) cudaGraphDestroy(graph);
    cudaFree(d_p);

    return iter;
}

// Explicit instantiations -----------------------------------------------------
template int run_signal_prop_impl<float>(
    cublasHandle_t, cudaStream_t, int, double,
    const float*, const float*, float*, float*, float*,
    int, double, int, float*);

template int run_signal_prop_impl<double>(
    cublasHandle_t, cudaStream_t, int, double,
    const double*, const double*, double*, double*, double*,
    int, double, int, double*);

template int run_signal_prop_impl<__half>(
    cublasHandle_t, cudaStream_t, int, double,
    const __half*, const __half*, __half*, __half*, __half*,
    int, double, int, __half*);

}  // namespace sfa
