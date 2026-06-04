// backends/cpu/kernels/signal/wavelets/ricker.cpp
// CPU kernel for Ricker (Mexican hat) wavelet.
// w[n] = (2/(sqrt(3*a)*pi^0.25)) * (1 - (t/a)^2) * exp(-t^2/(2*a^2))
// where t = n - (M-1)/2
// inputs[0]: a (width parameter, F64, 1-element)
// outputs[0]: real wavelet (F64)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status signal_ricker_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *a_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a_arr || !out) {
    cpu_set_last_error("signal_ricker: null array pointer");
    return C_FAILED;
  }

  if (!a_arr->data || !out->data) {
    cpu_set_last_error("signal_ricker: null data pointer");
    return C_FAILED;
  }

  double a = *(const double *)a_arr->data;
  int64_t M = out->numel;

  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    double half = (double)(M - 1) / 2.0;

    // Normalization constant: 2 / (sqrt(3*a) * pi^0.25)
    double norm = 2.0 / (std::sqrt(3.0 * a) * std::pow(M_PI, 0.25));
    double inv_a = 1.0 / a;
    double inv_a2 = inv_a * inv_a;

#ifdef _OPENMP
    if (M > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < M; ++i) {
        double t = (double)i - half;
        double ta = t * inv_a;
        data[i] = norm * (1.0 - ta * ta) * std::exp(-0.5 * t * t * inv_a2);
      }
    } else {
#endif
      for (int64_t i = 0; i < M; ++i) {
        double t = (double)i - half;
        double ta = t * inv_a;
        data[i] = norm * (1.0 - ta * ta) * std::exp(-0.5 * t * t * inv_a2);
      }
#ifdef _OPENMP
    }
#endif
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    float half = (float)(M - 1) / 2.0f;
    float a_f = (float)a;

    // Normalization constant: 2 / (sqrt(3*a) * pi^0.25)
    float norm = 2.0f / (std::sqrt(3.0f * a_f) * std::pow((float)M_PI, 0.25f));
    float inv_a = 1.0f / a_f;
    float inv_a2 = inv_a * inv_a;

#ifdef _OPENMP
    if (M > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < M; ++i) {
        float t = (float)i - half;
        float ta = t * inv_a;
        data[i] = norm * (1.0f - ta * ta) * std::exp(-0.5f * t * t * inv_a2);
      }
    } else {
#endif
      for (int64_t i = 0; i < M; ++i) {
        float t = (float)i - half;
        float ta = t * inv_a;
        data[i] = norm * (1.0f - ta * ta) * std::exp(-0.5f * t * t * inv_a2);
      }
#ifdef _OPENMP
    }
#endif
  } else {
    cpu_set_last_error("signal_ricker: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_ricker, INSIGHT_DTYPE_F64, signal_ricker_kernel_cpu);
REGISTER_CPU_KERNEL(signal_ricker, INSIGHT_DTYPE_F32, signal_ricker_kernel_cpu);
