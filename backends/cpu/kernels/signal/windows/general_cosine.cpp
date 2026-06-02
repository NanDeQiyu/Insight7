// backends/cpu/kernels/signal/windows/general_cosine.cpp
// CPU kernel for general cosine window generation.
// w[n] = sum_{k=0}^{K-1} a[k] * cos(k * (-pi + 2*pi*n/(M-1)))
// inputs[0]: coefficients array (F64, K elements)
// outputs[0]: window output
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status general_cosine_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *coeffs_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!coeffs_arr || !out) {
    cpu_set_last_error("general_cosine: null array pointer");
    return C_FAILED;
  }

  const double *a = (const double *)coeffs_arr->data;
  int64_t K = coeffs_arr->numel;
  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    if (M == 1) {
      // Single point: sum of all coefficients
      double sum = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        // cos(k * pi) = (-1)^k
        double sign = (k % 2 == 0) ? 1.0 : -1.0;
        sum += a[k] * sign;
      }
      data[0] = sum;
      return C_SUCCESS;
    }
    double base = -M_PI;
    double step = 2.0 * M_PI / (M - 1);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
    for (int64_t i = 0; i < M; ++i) {
      double angle = base + step * i;
      double val = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        val += a[k] * cos(k * angle);
      }
      data[i] = val;
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    if (M == 1) {
      // Single point: sum of all coefficients
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        float sign = (k % 2 == 0) ? 1.0f : -1.0f;
        sum += (float)a[k] * sign;
      }
      data[0] = sum;
      return C_SUCCESS;
    }
    float base = -(float)M_PI;
    float step = 2.0f * (float)M_PI / (M - 1);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
    for (int64_t i = 0; i < M; ++i) {
      float angle = base + step * i;
      float val = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        val += (float)a[k] * cosf(k * angle);
      }
      data[i] = val;
    }
  } else {
    cpu_set_last_error("general_cosine: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(general_cosine, INSIGHT_DTYPE_F64,
                    general_cosine_kernel_cpu);
REGISTER_CPU_KERNEL(general_cosine, INSIGHT_DTYPE_F32,
                    general_cosine_kernel_cpu);
