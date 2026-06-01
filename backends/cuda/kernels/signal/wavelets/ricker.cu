// backends/cuda/kernels/signal/wavelets/ricker.cu
// CUDA kernel for Ricker (Mexican hat) wavelet.
// w[n] = (2/(sqrt(3*a)*pi^0.25)) * (1 - (t/a)^2) * exp(-t^2/(2*a^2))
// where t = n - (M-1)/2
// inputs[0]: a (width parameter, F64, 1-element)
// outputs[0]: real wavelet (F64)
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void signal_ricker_kernel(double *out, int64_t M, double a) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;

  double half = (double)(M - 1) / 2.0;
  double t = (double)i - half;
  double inv_a = 1.0 / a;
  double ta = t * inv_a;
  double norm = 2.0 / (sqrt(3.0 * a) * pow(M_PI, 0.25));
  out[i] = norm * (1.0 - ta * ta) * exp(-0.5 * t * t * inv_a * inv_a);
}

extern "C" {

C_Status signal_ricker_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *a_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a_arr || !out) {
    gpu_set_last_error("signal_ricker: null array pointer");
    return C_FAILED;
  }

  if (!a_arr->data || !out->data) {
    gpu_set_last_error("signal_ricker: null data pointer");
    return C_FAILED;
  }

  double a;
  cudaMemcpy(&a, a_arr->data, sizeof(double), cudaMemcpyDeviceToHost);

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    signal_ricker_kernel<<<blocks, threads>>>((double *)out->data, M, a);
    break;
  default:
    gpu_set_last_error("signal_ricker: only F64 dtype supported");
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

REGISTER_GPU_KERNEL(signal_ricker, INSIGHT_DTYPE_F64, signal_ricker_kernel_gpu);
