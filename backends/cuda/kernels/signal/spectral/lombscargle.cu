// backends/cuda/kernels/signal/spectral/lombscargle.cu
// CUDA kernel for Lomb-Scargle periodogram.
// Each thread computes the power at one frequency.
// inputs[0]: x (sample times, F64)
// inputs[1]: y (sample values, F64)
// inputs[2]: freqs (evaluation frequencies, F64)
// outputs[0]: power spectral density (F64)
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void signal_lombscargle_kernel(double *p, const double *x,
                                          const double *y, const double *freqs,
                                          int64_t n, int64_t nf) {
  int64_t fi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (fi >= nf)
    return;

  double f = freqs[fi];
  double w = 2.0 * M_PI * f;

  if (w == 0.0) {
    double sum_y = 0.0;
    for (int64_t j = 0; j < n; ++j)
      sum_y += y[j];
    p[fi] = (sum_y * sum_y) / (double)(n * n);
    return;
  }

  // Compute tau
  double sin2wt = 0.0, cos2wt = 0.0;
  for (int64_t j = 0; j < n; ++j) {
    double arg = 2.0 * w * x[j];
    sin2wt += sin(arg);
    cos2wt += cos(arg);
  }
  double tau = atan2(sin2wt, cos2wt) / (2.0 * w);

  // Compute P
  double cs = 0.0, cc = 0.0, sn = 0.0, ss = 0.0;
  for (int64_t j = 0; j < n; ++j) {
    double arg = w * (x[j] - tau);
    double c = cos(arg);
    double s = sin(arg);
    cs += y[j] * c;
    cc += c * c;
    sn += y[j] * s;
    ss += s * s;
  }

  double pval = 0.0;
  if (cc > 1e-30)
    pval += (cs * cs) / cc;
  if (ss > 1e-30)
    pval += (sn * sn) / ss;
  p[fi] = 0.5 * pval;
}

extern "C" {

C_Status signal_lombscargle_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *y_arr = (InsightArray *)inputs[1];
  InsightArray *freqs_arr = (InsightArray *)inputs[2];
  InsightArray *p_arr = (InsightArray *)outputs[0];

  if (!x_arr || !y_arr || !freqs_arr || !p_arr) {
    gpu_set_last_error("signal_lombscargle: null array pointer");
    return C_FAILED;
  }

  if (!x_arr->data || !y_arr->data || !freqs_arr->data || !p_arr->data) {
    gpu_set_last_error("signal_lombscargle: null data pointer");
    return C_FAILED;
  }

  int64_t n = x_arr->numel;
  int64_t nf = freqs_arr->numel;
  if (nf == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (int)((nf + threads - 1) / threads);

  signal_lombscargle_kernel<<<blocks, threads>>>(
      (double *)p_arr->data, (const double *)x_arr->data,
      (const double *)y_arr->data, (const double *)freqs_arr->data, n, nf);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

} // extern "C"

REGISTER_GPU_KERNEL(signal_lombscargle, INSIGHT_DTYPE_F64,
                    signal_lombscargle_kernel_gpu);
