// backends/cuda/kernels/signal/waveforms/unit_impulse.cu
// CUDA kernel for unit impulse (Kronecker delta)
// w[n] = 1 if n == idx else 0
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

__global__ void unit_impulse_kernel_f64(double *out, int64_t idx, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  out[i] = (i == idx) ? 1.0 : 0.0;
}

__global__ void unit_impulse_kernel_f32(float *out, int64_t idx, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  out[i] = (i == idx) ? 1.0f : 0.0f;
}

extern "C" {

C_Status unit_impulse_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *idx_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!idx_arr || !out) {
    gpu_set_last_error("unit_impulse: null array pointer");
    return C_FAILED;
  }

  if (!idx_arr->data || idx_arr->numel < 1) {
    gpu_set_last_error("unit_impulse: idx array is empty");
    return C_FAILED;
  }

  int64_t idx = *(const int64_t *)idx_arr->data;
  int64_t M = out->numel;

  if (idx < 0 || idx >= M) {
    gpu_set_last_error("unit_impulse: idx out of range");
    return C_FAILED;
  }

  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  if (out->dtype == INSIGHT_DTYPE_F64) {
    unit_impulse_kernel_f64<<<blocks, threads>>>((double *)out->data, idx, M);
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    unit_impulse_kernel_f32<<<blocks, threads>>>((float *)out->data, idx, M);
  } else {
    gpu_set_last_error("unit_impulse: unsupported dtype, need F32 or F64");
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

REGISTER_GPU_KERNEL(unit_impulse, INSIGHT_DTYPE_F64, unit_impulse_kernel_gpu);
REGISTER_GPU_KERNEL(unit_impulse, INSIGHT_DTYPE_F32, unit_impulse_kernel_gpu);
