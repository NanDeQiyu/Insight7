// backends/cuda/kernels/signal/convolution/convolve1d.cu
// CUDA kernel for 1D direct convolution
// y[n] = sum_{k=0}^{nh-1} x[n+k-offset] * h[k]
// mode: 0=valid, 1=same, 2=full
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

__global__ void convolve1d_f64(const double *x, const double *h, double *y,
                               int64_t nx, int64_t nh, int64_t out_len,
                               int64_t offset) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_len)
    return;
  double sum = 0.0;
  for (int64_t j = 0; j < nh; ++j) {
    int64_t idx = i + j - offset;
    if (idx >= 0 && idx < nx) {
      sum += x[idx] * h[j];
    }
  }
  y[i] = sum;
}

__global__ void convolve1d_f32(const float *x, const float *h, float *y,
                               int64_t nx, int64_t nh, int64_t out_len,
                               int64_t offset) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_len)
    return;
  float sum = 0.0f;
  for (int64_t j = 0; j < nh; ++j) {
    int64_t idx = i + j - offset;
    if (idx >= 0 && idx < nx) {
      sum += x[idx] * h[j];
    }
  }
  y[i] = sum;
}

extern "C" {

C_Status convolve1d_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *h = (InsightArray *)inputs[1];
  int32_t mode = *(int32_t *)((InsightArray *)inputs[2])->data;
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !h || !out) {
    gpu_set_last_error("convolve1d: null array pointer");
    return C_FAILED;
  }

  int64_t nx = x->numel;
  int64_t nh = h->numel;
  int64_t out_len, offset;

  if (mode == 0) { // valid
    out_len = nx - nh + 1;
    offset = 0;
  } else if (mode == 1) { // same
    out_len = nx;
    offset = nh / 2;
  } else { // full
    out_len = nx + nh - 1;
    offset = 0;
  }

  int threads = 256;
  int blocks = (int)((out_len + threads - 1) / threads);

  if (x->dtype == INSIGHT_DTYPE_F64) {
    convolve1d_f64<<<blocks, threads>>>(
        (const double *)x->data, (const double *)h->data, (double *)out->data,
        nx, nh, out_len, offset);
  } else if (x->dtype == INSIGHT_DTYPE_F32) {
    convolve1d_f32<<<blocks, threads>>>(
        (const float *)x->data, (const float *)h->data, (float *)out->data, nx,
        nh, out_len, offset);
  } else {
    gpu_set_last_error("convolve1d: unsupported dtype, need F32 or F64");
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

REGISTER_GPU_KERNEL(convolve1d, INSIGHT_DTYPE_F32, convolve1d_kernel_gpu);
REGISTER_GPU_KERNEL(convolve1d, INSIGHT_DTYPE_F64, convolve1d_kernel_gpu);
