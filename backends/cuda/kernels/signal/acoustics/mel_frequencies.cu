// backends/cuda/kernels/signal/acoustics/mel_frequencies.cu
// CUDA kernel for generating mel frequency points.
// Generate n_mels equally spaced points in mel scale between f_low and f_high.
// inputs[0]: n_mels (I64, 1-element)
// inputs[1]: f_low (F64, 1-element)
// inputs[2]: f_high (F64, 1-element)
// outputs[0]: frequency array (F64)
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void signal_mel_frequencies_kernel(double *out, int64_t n_mels,
                                              double mel_low, double mel_step) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_mels)
    return;
  double mel_val = mel_low + (double)i * mel_step;
  out[i] = 700.0 * (pow(10.0, mel_val / 2595.0) - 1.0);
}

extern "C" {

C_Status signal_mel_frequencies_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *n_mels_arr = (InsightArray *)inputs[0];
  InsightArray *flo_arr = (InsightArray *)inputs[1];
  InsightArray *fhi_arr = (InsightArray *)inputs[2];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!n_mels_arr || !flo_arr || !fhi_arr || !out) {
    gpu_set_last_error("signal_mel_frequencies: null array pointer");
    return C_FAILED;
  }

  if (!n_mels_arr->data || !flo_arr->data || !fhi_arr->data || !out->data) {
    gpu_set_last_error("signal_mel_frequencies: null data pointer");
    return C_FAILED;
  }

  int64_t n_mels;
  double f_low, f_high;
  cudaMemcpy(&n_mels, n_mels_arr->data, sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&f_low, flo_arr->data, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&f_high, fhi_arr->data, sizeof(double), cudaMemcpyDeviceToHost);

  if (n_mels <= 0) {
    gpu_set_last_error("signal_mel_frequencies: n_mels must be positive");
    return C_FAILED;
  }

  double mel_low = 2595.0 * log10(1.0 + f_low / 700.0);
  double mel_high = 2595.0 * log10(1.0 + f_high / 700.0);
  double mel_step =
      (n_mels > 1) ? (mel_high - mel_low) / (double)(n_mels - 1) : 0.0;

  int threads = 256;
  int blocks = (int)((n_mels + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    signal_mel_frequencies_kernel<<<blocks, threads>>>(
        (double *)out->data, n_mels, mel_low, mel_step);
    break;
  default:
    gpu_set_last_error("signal_mel_frequencies: only F64 dtype supported");
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

REGISTER_GPU_KERNEL(signal_mel_frequencies, INSIGHT_DTYPE_F64,
                    signal_mel_frequencies_kernel_gpu);
