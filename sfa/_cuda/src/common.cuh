// Shared CUDA utility macros.
#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define SFA_CUDA_CHECK(expr)                                                  \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(_e) +                 \
                                     " (" #expr ")");                         \
        }                                                                     \
    } while (0)

#define SFA_CUBLAS_CHECK(expr)                                                \
    do {                                                                      \
        cublasStatus_t _s = (expr);                                           \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                    \
            throw std::runtime_error(std::string("cuBLAS error ") +           \
                                     std::to_string((int)_s) +                \
                                     " (" #expr ")");                         \
        }                                                                     \
    } while (0)
