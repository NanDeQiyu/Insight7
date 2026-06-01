// backends/cuda/kernels/signal/waveforms/sawtooth.cu
// CUDA kernel for sawtooth/triangle waveform
// w[n] = 2 * (t*n - floor(0.5 + t*n)) where t = width parameter
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void sawtooth_kernel_f64(double *out, double width, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  double tn = width * (double)i;
  out[i] = 2.0 * (tn - floor(0.5 + tn));
}

__global__ void sawtooth_kernel_f32(float *out, float width, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float tn = width * (float)i;
  out[i] = 2.0f * (tn - floorf(0.5f + tn));
}

extern "C" {

C_Status sawtooth_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *width_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!width_arr || !out) {
    gpu_set_last_error("sawtooth: null array pointer");
    return C_FAILED;
  }

  if (!width_arr->data || width_arr->numel < 1) {
    gpu_set_last_error("sawtooth: width array is empty");
    return C_FAILED;
  }

  double width = *(const double *)width_arr->data;
  int64_t M = out->numel;

  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  if (out->dtype == INSIGHT_DTYPE_F64) {
    sawtooth_kernel_f64<<<blocks, threads>>>((double *)out->data, width, M);
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    sawtooth_kernel_f32<<<blocks, threads>>>((float *)out->data, (float)width,
                                             M);
  } else {
    gpu_set_last_error("sawtooth: unsupported dtype, need F32 or F64");
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

REGISTER_GPU_KERNEL(sawtooth, INSIGHT_DTYPE_F64, sawtooth_kernel_gpu);
REGISTER_GPU_KERNEL(sawtooth, INSIGHT_DTYPE_F32, sawtooth_kernel_gpu);
