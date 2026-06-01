// backends/cuda/kernels/signal/windows/triang.cu
// CUDA kernel for triangular window generation.
// Matches scipy.signal.windows.triang implementation.
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_runtime.h>

template <typename T> __global__ void triang_kernel_odd(T *out, int64_t M) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= M)
    return;
  T denom = T(M / 2 + 1);
  if (n <= (M - 1) / 2) {
    out[n] = T(n + 1) / denom;
  } else {
    out[n] = T(M - n) / denom;
  }
}

template <typename T> __global__ void triang_kernel_even(T *out, int64_t M) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= M)
    return;
  T denom = T(M / 2);
  if (n < M / 2) {
    out[n] = T(n + 1) / denom;
  } else {
    out[n] = T(M - n) / denom;
  }
}

extern "C" {

C_Status triang_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    gpu_set_last_error("triang: null output pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  dim3 blocks((int)((M + 255) / 256));
  dim3 threads(256);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    if (M % 2 == 1) {
      triang_kernel_odd<double><<<blocks, threads>>>((double *)out->data, M);
    } else {
      triang_kernel_even<double><<<blocks, threads>>>((double *)out->data, M);
    }
    break;
  default:
    gpu_set_last_error("triang: only F64 dtype supported");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

} // extern "C"

REGISTER_GPU_KERNEL(triang, INSIGHT_DTYPE_F64, triang_kernel_gpu);
