// backends/cuda/kernels/signal/filtering/detrend.cu
// CUDA kernel for linear detrending.
// Fits y = a + b*x via least squares, subtracts trend from data.
// inputs[0]: data array (F64 or F32)
// outputs[0]: detrended array
//
// Note: The slope/intercept computation uses a two-pass approach:
// first compute sums on host (fast for single array), then subtract
// trend on GPU. For very large arrays, a reduction kernel would be
// more efficient, but the sums are O(n) with no parallelism issues.
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

template <typename T>
__global__ void signal_detrend_kernel(T *out, const T *x, int64_t n, T a, T b) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  out[i] = x[i] - (a + b * (T)i);
}

extern "C" {

C_Status signal_detrend_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *y_arr = (InsightArray *)outputs[0];

  if (!x_arr || !y_arr) {
    gpu_set_last_error("signal_detrend: null array pointer");
    return C_FAILED;
  }

  if (!x_arr->data || !y_arr->data) {
    gpu_set_last_error("signal_detrend: null data pointer");
    return C_FAILED;
  }

  int64_t n = x_arr->numel;
  if (n == 0)
    return C_SUCCESS;

  // Copy data to host for sum computation
  double sum_x = (double)n * (double)(n - 1) * 0.5;
  double sum_x2 = (double)(n - 1) * (double)n * (double)(2 * n - 1) / 6.0;
  double nd = (double)n;
  double denom = nd * sum_x2 - sum_x * sum_x;

  double sum_y = 0.0, sum_xy = 0.0;

  switch (x_arr->dtype) {
  case INSIGHT_DTYPE_F64: {
    // Copy to host for sum computation
    std::vector<double> host_x(n);
    cudaMemcpy(host_x.data(), x_arr->data, n * sizeof(double),
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < n; ++i) {
      sum_y += host_x[i];
      sum_xy += (double)i * host_x[i];
    }

    double a = 0.0, b = 0.0;
    if (std::abs(denom) > 1e-30) {
      a = (sum_y * sum_x2 - sum_x * sum_xy) / denom;
      b = (nd * sum_xy - sum_x * sum_y) / denom;
    } else {
      a = sum_y / nd;
    }

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    signal_detrend_kernel<double><<<blocks, threads>>>(
        (double *)y_arr->data, (const double *)x_arr->data, n, a, b);
    break;
  }
  case INSIGHT_DTYPE_F32: {
    std::vector<float> host_x(n);
    cudaMemcpy(host_x.data(), x_arr->data, n * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < n; ++i) {
      sum_y += (double)host_x[i];
      sum_xy += (double)i * (double)host_x[i];
    }

    double a = 0.0, b = 0.0;
    if (std::abs(denom) > 1e-30) {
      a = (sum_y * sum_x2 - sum_x * sum_xy) / denom;
      b = (nd * sum_xy - sum_x * sum_y) / denom;
    } else {
      a = sum_y / nd;
    }

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    signal_detrend_kernel<float>
        <<<blocks, threads>>>((float *)y_arr->data, (const float *)x_arr->data,
                              n, (float)a, (float)b);
    break;
  }
  default:
    gpu_set_last_error("signal_detrend: unsupported dtype, need F32 or F64");
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

REGISTER_GPU_KERNEL(signal_detrend, INSIGHT_DTYPE_F64,
                    signal_detrend_kernel_gpu);
REGISTER_GPU_KERNEL(signal_detrend, INSIGHT_DTYPE_F32,
                    signal_detrend_kernel_gpu);
