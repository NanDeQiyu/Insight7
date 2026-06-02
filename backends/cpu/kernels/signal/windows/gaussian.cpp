// backends/cpu/kernels/signal/windows/gaussian.cpp
// CPU kernel for Gaussian window generation.
// w[n] = exp(-0.5 * ((n - (M-1)/2) / (sigma * (M-1)/2))^2)
// inputs[0]: sigma (1-element F64 array)
// outputs[0]: window output
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status gaussian_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *sigma_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!sigma_arr || !out) {
    cpu_set_last_error("gaussian: null array pointer");
    return C_FAILED;
  }

  double sigma = *(double *)sigma_arr->data;
  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    double half = (M - 1) / 2.0;
    double denom = sigma * half;
    if (denom == 0.0) {
      // All centered at peak
      for (int64_t i = 0; i < M; ++i) {
        data[i] = (i == (int64_t)half) ? 1.0 : 0.0;
      }
    } else {
      double scale = -0.5 / (denom * denom);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
      for (int64_t i = 0; i < M; ++i) {
        double diff = i - half;
        data[i] = exp(scale * diff * diff);
      }
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    float sigma_f = (float)sigma;
    float half = (M - 1) / 2.0f;
    float denom = sigma_f * half;
    if (denom == 0.0f) {
      // All centered at peak
      for (int64_t i = 0; i < M; ++i) {
        data[i] = (i == (int64_t)half) ? 1.0f : 0.0f;
      }
    } else {
      float scale = -0.5f / (denom * denom);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
      for (int64_t i = 0; i < M; ++i) {
        float diff = i - half;
        data[i] = expf(scale * diff * diff);
      }
    }
  } else {
    cpu_set_last_error("gaussian: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(gaussian, INSIGHT_DTYPE_F64, gaussian_kernel_cpu);
REGISTER_CPU_KERNEL(gaussian, INSIGHT_DTYPE_F32, gaussian_kernel_cpu);
