// backends/cuda/kernels/signal/acoustics/hz2bark.cu
// CUDA kernel for Hz to Bark conversion.
// bark = 13 * atan(0.00076 * hz) + 3.5 * atan((hz/7500)^2)
// inputs[0]: hz array (F64 or F32)
// outputs[0]: bark array (same type)
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__global__ void signal_hz2bark_kernel(T *out, const T *in, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  T ratio = in[i] / T(7500.0);
  out[i] = T(13.0) * atan(T(0.00076) * in[i]) + T(3.5) * atan(ratio * ratio);
}

extern "C" {

C_Status signal_hz2bark_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *hz_arr = (InsightArray *)inputs[0];
  InsightArray *bark_arr = (InsightArray *)outputs[0];

  if (!hz_arr || !bark_arr) {
    gpu_set_last_error("signal_hz2bark: null array pointer");
    return C_FAILED;
  }

  if (!hz_arr->data || !bark_arr->data) {
    gpu_set_last_error("signal_hz2bark: null data pointer");
    return C_FAILED;
  }

  int64_t n = hz_arr->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);

  switch (hz_arr->dtype) {
  case INSIGHT_DTYPE_F64:
    signal_hz2bark_kernel<double><<<blocks, threads>>>(
        (double *)bark_arr->data, (const double *)hz_arr->data, n);
    break;
  case INSIGHT_DTYPE_F32:
    signal_hz2bark_kernel<float><<<blocks, threads>>>(
        (float *)bark_arr->data, (const float *)hz_arr->data, n);
    break;
  default:
    gpu_set_last_error("signal_hz2bark: unsupported dtype, need F32 or F64");
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

REGISTER_GPU_KERNEL(signal_hz2bark, INSIGHT_DTYPE_F64,
                    signal_hz2bark_kernel_gpu);
REGISTER_GPU_KERNEL(signal_hz2bark, INSIGHT_DTYPE_F32,
                    signal_hz2bark_kernel_gpu);
