// backends/cuda/kernels/signal/filter_design/firwin.cu
// CUDA kernel for FIR filter windowed sinc design.
// h[n] = sinc(2*fc*(n - (M-1)/2)) * window[n] where fc is cutoff frequency
// The window function is a Hann window applied element-wise.
// inputs[0]: fc (cutoff frequency, F64, 1-element)
// inputs[1]: M (filter length, I64, 1-element)
// outputs[0]: FIR coefficients (F64)
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ double sinc_dev(double x) {
  if (fabs(x) < 1e-15)
    return 1.0;
  return sin(M_PI * x) / (M_PI * x);
}

__global__ void signal_firwin_kernel(double *out, int64_t M_len, double fc) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M_len)
    return;

  double half = (double)(M_len - 1) / 2.0;
  double n_centered = (double)i - half;
  double h = sinc_dev(2.0 * fc * n_centered);
  double win = 0.5 * (1.0 - cos(2.0 * M_PI * (double)i / (double)(M_len - 1)));
  out[i] = h * win;
}

extern "C" {

C_Status signal_firwin_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *fc_arr = (InsightArray *)inputs[0];
  InsightArray *M_arr = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!fc_arr || !M_arr || !out) {
    gpu_set_last_error("signal_firwin: null array pointer");
    return C_FAILED;
  }

  if (!fc_arr->data || !M_arr->data || !out->data) {
    gpu_set_last_error("signal_firwin: null data pointer");
    return C_FAILED;
  }

  double fc;
  int64_t M_len;
  cudaMemcpy(&fc, fc_arr->data, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&M_len, M_arr->data, sizeof(int64_t), cudaMemcpyDeviceToHost);

  if (M_len <= 0) {
    gpu_set_last_error("signal_firwin: M must be positive");
    return C_FAILED;
  }

  int threads = 256;
  int blocks = (int)((M_len + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    signal_firwin_kernel<<<blocks, threads>>>((double *)out->data, M_len, fc);
    break;
  default:
    gpu_set_last_error("signal_firwin: only F64 dtype supported");
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

REGISTER_GPU_KERNEL(signal_firwin, INSIGHT_DTYPE_F64, signal_firwin_kernel_gpu);
