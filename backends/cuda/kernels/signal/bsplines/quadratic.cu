// backends/cuda/kernels/signal/bsplines/quadratic.cu
// CUDA kernel for quadratic B-spline basis function
// |x| < 0.5:     3/4 - x^2
// 0.5 <= |x| < 1.5: (3-2*|x|)^2 / 8
// |x| >= 1.5:    0
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void quadratic_kernel_f64(double *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  double ax = fabs(data[i]);
  if (ax < 0.5) {
    data[i] = 0.75 - ax * ax;
  } else if (ax < 1.5) {
    double t = 3.0 - 2.0 * ax;
    data[i] = (t * t) / 8.0;
  } else {
    data[i] = 0.0;
  }
}

__global__ void quadratic_kernel_f32(float *data, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float ax = fabsf(data[i]);
  if (ax < 0.5f) {
    data[i] = 0.75f - ax * ax;
  } else if (ax < 1.5f) {
    float t = 3.0f - 2.0f * ax;
    data[i] = (t * t) / 8.0f;
  } else {
    data[i] = 0.0f;
  }
}

extern "C" {

C_Status quadratic_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    gpu_set_last_error("quadratic: null array pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  if (out->dtype == INSIGHT_DTYPE_F64) {
    quadratic_kernel_f64<<<blocks, threads>>>((double *)out->data, M);
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    quadratic_kernel_f32<<<blocks, threads>>>((float *)out->data, M);
  } else {
    gpu_set_last_error("quadratic: unsupported dtype, need F32 or F64");
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

REGISTER_GPU_KERNEL(quadratic, INSIGHT_DTYPE_F64, quadratic_kernel_gpu);
REGISTER_GPU_KERNEL(quadratic, INSIGHT_DTYPE_F32, quadratic_kernel_gpu);
