// backends/cuda/kernels/signal/windows/general_cosine.cu
// CUDA kernel for general cosine window generation.
// w[n] = sum_{k=0}^{K-1} a[k] * cos(k * (-pi + 2*pi*n/(M-1)))
// inputs[0]: coefficients array (F64, K elements on device)
// outputs[0]: window output
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_runtime.h>

template <typename T>
__global__ void general_cosine_kernel(T *out, const T *a, int64_t M,
                                      int64_t K) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  T angle = -T(M_PI) + T(2.0 * M_PI) * T(i) / T(M - 1);
  T val = T(0.0);
  for (int64_t k = 0; k < K; ++k) {
    val += a[k] * cos(T(k) * angle);
  }
  out[i] = val;
}

template <typename T>
__global__ void general_cosine_single_point(T *out, const T *a, int64_t K) {
  // For M==1: sum of a[k]*cos(k*pi) = sum of a[k]*(-1)^k
  T val = T(0.0);
  for (int64_t k = 0; k < K; ++k) {
    T sign = (k % 2 == 0) ? T(1.0) : T(-1.0);
    val += a[k] * sign;
  }
  out[0] = val;
}

extern "C" {

C_Status general_cosine_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *coeffs_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!coeffs_arr || !out) {
    gpu_set_last_error("general_cosine: null array pointer");
    return C_FAILED;
  }

  int64_t K = coeffs_arr->numel;
  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64: {
    if (M == 1) {
      general_cosine_single_point<double>
          <<<1, 1>>>((double *)out->data, (const double *)coeffs_arr->data, K);
    } else {
      dim3 blocks((int)((M + 255) / 256));
      dim3 threads(256);
      general_cosine_kernel<double><<<blocks, threads>>>(
          (double *)out->data, (const double *)coeffs_arr->data, M, K);
    }
    break;
  }
  default:
    gpu_set_last_error("general_cosine: only F64 dtype supported");
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

REGISTER_GPU_KERNEL(general_cosine, INSIGHT_DTYPE_F64,
                    general_cosine_kernel_gpu);
