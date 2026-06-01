// backends/cuda/kernels/signal/windows/hamming.cu
// CUDA kernel for Hamming window generation.
// w[n] = 0.54 - 0.46 * cos(2*pi*n/(M-1))
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_runtime.h>

template <typename T> __global__ void hamming_kernel(T *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  if (M == 1) {
    out[0] = T(1.0);
    return;
  }
  out[i] = T(0.54) - T(0.46) * cos(T(2.0 * M_PI) * T(i) / T(M - 1));
}

extern "C" {

C_Status hamming_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    gpu_set_last_error("hamming: null output pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  dim3 blocks((int)((M + 255) / 256));
  dim3 threads(256);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    hamming_kernel<double><<<blocks, threads>>>((double *)out->data, M);
    break;
  default:
    gpu_set_last_error("hamming: only F64 dtype supported");
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

REGISTER_GPU_KERNEL(hamming, INSIGHT_DTYPE_F64, hamming_kernel_gpu);
