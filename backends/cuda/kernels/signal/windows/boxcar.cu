// backends/cuda/kernels/signal/windows/boxcar.cu
// CUDA kernel for boxcar (rectangular) window generation.
// w[n] = 1.0 for all n
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T> __global__ void boxcar_kernel(T *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  out[i] = T(1.0);
}

extern "C" {

C_Status boxcar_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    gpu_set_last_error("boxcar: null output pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  dim3 blocks((int)((M + 255) / 256));
  dim3 threads(256);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    boxcar_kernel<double><<<blocks, threads>>>((double *)out->data, M);
    break;
  default:
    gpu_set_last_error("boxcar: only F64 dtype supported");
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

REGISTER_GPU_KERNEL(boxcar, INSIGHT_DTYPE_F64, boxcar_kernel_gpu);
