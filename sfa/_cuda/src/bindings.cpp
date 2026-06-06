// pybind11 entry point for the native CUDA kernels of sfa 0.2.x.
//
// Layout convention
// -----------------
// numpy arrays arrive as row-major (C-contiguous). cuBLAS is column-major.
// We never explicitly transpose: every NxN buffer on the device holds the
// same bytes as the row-major numpy array, and the column-major view of
// those bytes is the mathematical transpose of the matrix. The GEMM in
// influence_iter.cu issues the column-major call  W_T * S_T = (S*W)_T,
// whose column-major bytes equal the row-major bytes of S*W.
//
// dtype support
// -------------
// The Python caller selects the compute dtype via the `dtype` parameter:
//   "float32"        -> SGEMM (+ optional TF32 Tensor Core math)
//   "float64"        -> DGEMM  (slow on consumer Ada, exact)
//   "float16"        -> cuBLAS GemmEx FP16-in / FP32-accumulate (HMMA)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "common.cuh"
#include "dtype.cuh"

namespace py = pybind11;

namespace sfa {

// Declarations of templated impls instantiated in influence_iter.cu.
template <class T>
int run_influence_iteration_impl(
    cublasHandle_t, cudaStream_t, int N, double alpha,
    const T* d_W, T* d_S0, T* d_scratch,
    T* d_diff,
    int max_iter, double tol, int check_every);

template <class T>
void scale_inplace_impl(cudaStream_t stream, T* d_S, double beta, size_t n);

// --- RAII helpers -----------------------------------------------------------
struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t b) { alloc(b); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept { *this = std::move(o); }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr) cudaFree(ptr);
            ptr = o.ptr; bytes = o.bytes;
            o.ptr = nullptr; o.bytes = 0;
        }
        return *this;
    }
    void alloc(size_t b) { SFA_CUDA_CHECK(cudaMalloc(&ptr, b)); bytes = b; }
    template <class U> U* as() { return reinterpret_cast<U*>(ptr); }
};

struct CublasHandle {
    cublasHandle_t h = nullptr;
    CublasHandle() { SFA_CUBLAS_CHECK(cublasCreate(&h)); }
    ~CublasHandle() { if (h) cublasDestroy(h); }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
};

struct Stream {
    cudaStream_t s = nullptr;
    Stream() { SFA_CUDA_CHECK(cudaStreamCreate(&s)); }
    ~Stream() { if (s) cudaStreamDestroy(s); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

// --- Diagonal-set helper ----------------------------------------------------
// S <- I (NxN). The buffer is zeroed via cudaMemset, then the diagonal is set
// to 1 element of T using a strided 2D memcpy from a host vector of ones.
template <class T>
static void initialize_identity(T* d_S, int N) {
    SFA_CUDA_CHECK(cudaMemset(d_S, 0, sizeof(T) * (size_t)N * N));
    std::vector<T> ones((size_t)N);
    if constexpr (std::is_same_v<T, __half>) {
        for (int i = 0; i < N; ++i) ones[i] = __float2half(1.0f);
    } else {
        for (int i = 0; i < N; ++i) ones[i] = T(1);
    }
    SFA_CUDA_CHECK(cudaMemcpy2D(
        d_S, sizeof(T) * (size_t)(N + 1),
        ones.data(), sizeof(T),
        sizeof(T), (size_t)N,
        cudaMemcpyHostToDevice));
}

// --- Per-dtype runner -------------------------------------------------------
template <class T>
static py::tuple run_with_dtype(
        py::array W_np,
        double alpha, double beta,
        int max_iter, double tol,
        int device_id, int check_every,
        bool use_tf32)
{
    // numpy is row-major. We expect a NxN C-contiguous array of T.
    auto buf = W_np.request();
    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
        throw std::invalid_argument("W must be a square 2D array.");
    }
    if (buf.itemsize != sizeof(T)) {
        throw std::invalid_argument("W dtype mismatch (size).");
    }
    const int N = (int)buf.shape[0];
    const size_t elems = (size_t)N * N;
    const size_t bytes = elems * sizeof(T);

    SFA_CUDA_CHECK(cudaSetDevice(device_id));

    DeviceBuffer d_W(bytes);
    SFA_CUDA_CHECK(cudaMemcpy(d_W.ptr, buf.ptr, bytes, cudaMemcpyHostToDevice));

    Stream stream;
    CublasHandle cublas;
    SFA_CUBLAS_CHECK(cublasSetStream(cublas.h, stream.s));

    // TF32 math mode only meaningfully applies to the FP32 path; on the FP64
    // path it is a no-op, on FP16 cuBLAS uses Tensor Cores regardless.
    SFA_CUBLAS_CHECK(cublasSetMathMode(
        cublas.h,
        use_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));

    DeviceBuffer d_S(bytes);
    DeviceBuffer d_scratch(2 * bytes);
    DeviceBuffer d_diff(sizeof(T));   // strict: accumulator in user's dtype

    initialize_identity<T>(d_S.as<T>(), N);

    int num_iter = run_influence_iteration_impl<T>(
        cublas.h, stream.s,
        N, alpha,
        d_W.as<T>(),
        d_S.as<T>(),
        d_scratch.as<T>(),
        d_diff.as<T>(),
        max_iter, tol, check_every);

    if (beta != 1.0) {
        scale_inplace_impl<T>(stream.s, d_S.as<T>(), beta, elems);
    }
    SFA_CUDA_CHECK(cudaStreamSynchronize(stream.s));

    // Allocate an output array of the correct dtype.
    py::array out = py::array(W_np.dtype(), {N, N});
    SFA_CUDA_CHECK(cudaMemcpy(out.mutable_data(), d_S.ptr,
                              bytes, cudaMemcpyDeviceToHost));

    return py::make_tuple(std::move(out), num_iter);
}

// Ensure the array is C-contiguous (copy if needed) without changing dtype.
// We rely on the Python wrapper to have already cast W to the requested dtype
// via numpy.ascontiguousarray(W, dtype=...). This avoids pybind11's
// `py::array_t<uintN_t, forcecast>` trick that would reinterpret bytes and
// destroy float16 bit patterns.
static py::array as_c_contig_same_dtype(py::array W_np) {
    py::array w = py::array::ensure(W_np, py::array::c_style);
    if (!w) throw std::runtime_error("W could not be made C-contiguous");
    return w;
}

static void check_dtype_match(const py::array& W, size_t expected_itemsize,
                              const char* name) {
    if ((size_t)W.dtype().itemsize() != expected_itemsize) {
        throw std::invalid_argument(
            std::string(name) + ": dtype itemsize mismatch (got " +
            std::to_string(W.dtype().itemsize()) + ", expected " +
            std::to_string(expected_itemsize) + ")");
    }
}

// --- Public entry: dtype dispatch ------------------------------------------
py::tuple compute_influence_cuda(
        py::array W_np,
        double alpha,
        double beta,
        int    max_iter,
        double tol,
        int    device_id,
        int    check_every,
        bool   use_tf32,
        const  std::string& dtype)
{
    py::array W = as_c_contig_same_dtype(W_np);
    if (dtype == "float32" || dtype == "f32") {
        check_dtype_match(W, sizeof(float), "W");
        return run_with_dtype<float>(W, alpha, beta, max_iter, tol,
                                     device_id, check_every, use_tf32);
    } else if (dtype == "float64" || dtype == "f64") {
        check_dtype_match(W, sizeof(double), "W");
        return run_with_dtype<double>(W, alpha, beta, max_iter, tol,
                                      device_id, check_every, use_tf32);
    } else if (dtype == "float16" || dtype == "f16" || dtype == "half") {
        check_dtype_match(W, sizeof(__half), "W");
        return run_with_dtype<__half>(W, alpha, beta, max_iter, tol,
                                      device_id, check_every, use_tf32);
    }
    throw std::invalid_argument(
        "Unsupported dtype: " + dtype +
        " (supported: float32, float64, float16)");
}

// --- Device introspection --------------------------------------------------
py::dict device_info(int device_id)
{
    cudaDeviceProp prop{};
    SFA_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    py::dict d;
    d["name"]        = std::string(prop.name);
    d["compute_cap"] = py::make_tuple(prop.major, prop.minor);
    d["total_mem"]   = (size_t)prop.totalGlobalMem;
    d["multi_proc"]  = prop.multiProcessorCount;
    d["max_threads"] = prop.maxThreadsPerBlock;
    return d;
}

int device_count()
{
    int n = 0;
    SFA_CUDA_CHECK(cudaGetDeviceCount(&n));
    return n;
}

// =====================================================================
// Forward declarations from signal_prop_iter.cu
// =====================================================================
template <class T>
int run_signal_prop_impl(
    cublasHandle_t, cudaStream_t, int N,
    double alpha,
    const T* d_W, const T* d_b,
    T* d_x_io, T* d_x_other,
    T* d_diff,
    int max_iter, double tol, int check_every,
    T* d_trj);

template <class T>
static py::tuple run_signal_prop_with_dtype(
        py::array W_np,
        py::array x0_np,
        py::array b_np,
        double alpha,
        int    max_iter,
        double tol,
        int    device_id,
        int    check_every,
        bool   use_tf32,
        bool   return_trajectory)
{
    auto wb = W_np.request();
    auto xb = x0_np.request();
    auto bb = b_np.request();
    if (wb.ndim != 2 || wb.shape[0] != wb.shape[1])
        throw std::invalid_argument("W must be NxN.");
    if (xb.ndim != 1 || xb.shape[0] != wb.shape[0])
        throw std::invalid_argument("x0 length must match N.");
    if (bb.ndim != 1 || bb.shape[0] != wb.shape[0])
        throw std::invalid_argument("b length must match N.");
    if (wb.itemsize != sizeof(T) || xb.itemsize != sizeof(T) ||
        bb.itemsize != sizeof(T))
        throw std::invalid_argument("dtype mismatch.");

    const int N = (int)wb.shape[0];
    const size_t mat_bytes = (size_t)N * N * sizeof(T);
    const size_t vec_bytes = (size_t)N * sizeof(T);

    SFA_CUDA_CHECK(cudaSetDevice(device_id));

    DeviceBuffer d_W(mat_bytes);
    DeviceBuffer d_b(vec_bytes);
    DeviceBuffer d_x(vec_bytes);
    DeviceBuffer d_x2(vec_bytes);
    DeviceBuffer d_diff(sizeof(T));   // strict-dtype accumulator
    SFA_CUDA_CHECK(cudaMemcpy(d_W.ptr, wb.ptr, mat_bytes, cudaMemcpyHostToDevice));
    SFA_CUDA_CHECK(cudaMemcpy(d_b.ptr, bb.ptr, vec_bytes, cudaMemcpyHostToDevice));
    SFA_CUDA_CHECK(cudaMemcpy(d_x.ptr, xb.ptr, vec_bytes, cudaMemcpyHostToDevice));

    // Trajectory buffer: shape (max_iter + 1, N). Allocated only if needed.
    DeviceBuffer d_trj;
    if (return_trajectory) {
        d_trj.alloc((size_t)(max_iter + 1) * (size_t)N * sizeof(T));
    }

    Stream stream;
    CublasHandle cublas;
    SFA_CUBLAS_CHECK(cublasSetStream(cublas.h, stream.s));
    SFA_CUBLAS_CHECK(cublasSetMathMode(
        cublas.h,
        use_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));

    T* d_trj_ptr = return_trajectory ? d_trj.as<T>() : nullptr;
    int num_iter = run_signal_prop_impl<T>(
        cublas.h, stream.s, N, alpha,
        d_W.as<T>(), d_b.as<T>(),
        d_x.as<T>(), d_x2.as<T>(),
        d_diff.as<T>(),
        max_iter, tol, check_every,
        d_trj_ptr);

    SFA_CUDA_CHECK(cudaStreamSynchronize(stream.s));

    py::array x_out = py::array(W_np.dtype(),
                                std::vector<py::ssize_t>{(py::ssize_t)N});
    SFA_CUDA_CHECK(cudaMemcpy(x_out.mutable_data(), d_x.ptr,
                              vec_bytes, cudaMemcpyDeviceToHost));

    if (return_trajectory) {
        // Download only the used rows: shape (num_iter + 1, N). Single
        // contiguous H2D transfer; the trailing tail of d_trj is unused but
        // never copied.
        py::array trj_out = py::array(
            W_np.dtype(),
            std::vector<py::ssize_t>{(py::ssize_t)(num_iter + 1),
                                     (py::ssize_t)N});
        SFA_CUDA_CHECK(cudaMemcpy(
            trj_out.mutable_data(), d_trj.ptr,
            (size_t)(num_iter + 1) * (size_t)N * sizeof(T),
            cudaMemcpyDeviceToHost));
        return py::make_tuple(std::move(x_out), std::move(trj_out));
    }
    return py::make_tuple(std::move(x_out), num_iter);
}

py::tuple compute_signal_propagation_cuda(
        py::array W_np,
        py::array x0_np,
        py::array b_np,
        double alpha,
        int    max_iter,
        double tol,
        int    device_id,
        int    check_every,
        bool   use_tf32,
        const  std::string& dtype,
        bool   return_trajectory)
{
    py::array W = as_c_contig_same_dtype(W_np);
    py::array x = as_c_contig_same_dtype(x0_np);
    py::array b = as_c_contig_same_dtype(b_np);
    if (dtype == "float32" || dtype == "f32") {
        check_dtype_match(W, sizeof(float),  "W");
        check_dtype_match(x, sizeof(float),  "x0");
        check_dtype_match(b, sizeof(float),  "b");
        return run_signal_prop_with_dtype<float>(
            W, x, b, alpha, max_iter, tol,
            device_id, check_every, use_tf32, return_trajectory);
    } else if (dtype == "float64" || dtype == "f64") {
        check_dtype_match(W, sizeof(double), "W");
        check_dtype_match(x, sizeof(double), "x0");
        check_dtype_match(b, sizeof(double), "b");
        return run_signal_prop_with_dtype<double>(
            W, x, b, alpha, max_iter, tol,
            device_id, check_every, use_tf32, return_trajectory);
    } else if (dtype == "float16" || dtype == "f16" || dtype == "half") {
        check_dtype_match(W, sizeof(__half), "W");
        check_dtype_match(x, sizeof(__half), "x0");
        check_dtype_match(b, sizeof(__half), "b");
        return run_signal_prop_with_dtype<__half>(
            W, x, b, alpha, max_iter, tol,
            device_id, check_every, use_tf32, return_trajectory);
    }
    throw std::invalid_argument(
        "Unsupported dtype: " + dtype +
        " (supported: float32, float64, float16)");
}

}  // namespace sfa

PYBIND11_MODULE(_native, m) {
    m.doc() = "sfa 0.2.x native CUDA kernels for NVIDIA Ada (sm_89).";

    m.def("compute_influence_cuda", &sfa::compute_influence_cuda,
          py::arg("W"),
          py::arg("alpha"),
          py::arg("beta") = 0.5,
          py::arg("max_iter") = 1000,
          py::arg("tol") = 1e-6,
          py::arg("device_id") = 0,
          py::arg("check_every") = 8,
          py::arg("use_tf32") = true,
          py::arg("dtype") = std::string("float32"),
          R"doc(Iterate S(t+1) = S(t)*(alpha*W) + I from S(0)=I on the GPU.
The compute dtype is selected by `dtype` ('float32', 'float64', 'float16').
Returns (beta * S_converged, num_iter).)doc");

    m.def("compute_signal_propagation_cuda", &sfa::compute_signal_propagation_cuda,
          py::arg("W"),
          py::arg("x0"),
          py::arg("b"),
          py::arg("alpha") = 0.5,
          py::arg("max_iter") = 1000,
          py::arg("tol") = 1e-5,
          py::arg("device_id") = 0,
          py::arg("check_every") = 8,
          py::arg("use_tf32") = true,
          py::arg("dtype") = std::string("float32"),
          py::arg("return_trajectory") = false,
          R"doc(Iterate x(t+1) = alpha*W*x(t) + (1-alpha)*b on the GPU.
The compute dtype is selected by `dtype` ('float32', 'float64', 'float16').

If ``return_trajectory`` is False (default), returns ``(x_converged, num_iter)``.
If True, returns ``(x_converged, trajectory)`` where ``trajectory`` has shape
``(num_iter + 1, N)`` with ``trajectory[0]`` equal to the initial state.
The trajectory mode disables CUDA Graph capture (per-iter row write) but
issues a single H2D download at the end, sized to the actual iter count.)doc");

    m.def("device_info",  &sfa::device_info,  py::arg("device_id") = 0);
    m.def("device_count", &sfa::device_count);
}
