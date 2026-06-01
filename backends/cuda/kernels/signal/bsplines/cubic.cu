// backends/cuda/kernels/signal/bsplines/cubic.cu
// CUDA kernel for cubic B-spline basis function
// |x| < 1:       2/3 - |x|^2*(2-|x|)/2
// 1 <= |x| < 2:  (2-|x|)^3 / 6
// |x| >= 2:      0
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void cubic_kernel_f64(double *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  double ax = fabs(data[i]);
  if (ax < 1.0) {
    double ax2 = ax * ax;
    data[i] = (2.0 / 3.0) - 0.5 * ax2 * (2.0 - ax);
  } else if (ax < 2.0) {
    double t = 2.0 - ax;
    data[i] = (t * t * t) / 6.0;
  } else {
    data[i] = 0.0;
  }
}

__global__ void cubic_kernel_f32(float *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float ax = fabsf(data[i]);
  if (ax < 1.0f) {
    float ax2 = ax * ax;
    data[i] = (2.0f / 3.0f) - 0.5f * ax2 * (2.0f - ax);
  } else if (ax < 2.0f) {
    float t = 2.0f - ax;
    data[i] = (t * t * t) / 6.0f;
  } else {
    data[i] = 0.0f;
  }
}

extern "C" {

C_Status cubic_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    gpu_set_last_error("cubic: null array pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  if (out->dtype == INSIGHT_DTYPE_F64) {
    cubic_kernel_f64<<<blocks, threads>>>((double *)out->data, M);
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    cubic_kernel_f32<<<blocks, threads>>>((float *)out->data, M);
  } else {
    gpu_set_last_error("cubic: unsupported dtype, need F32 or F64");
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

REGISTER_GPU_KERNEL(cubic, INSIGHT_DTYPE_F64, cubic_kernel_gpu);
REGISTER_GPU_KERNEL(cubic, INSIGHT_DTYPE_F32, cubic_kernel_gpu);
