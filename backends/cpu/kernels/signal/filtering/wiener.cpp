// backends/cpu/kernels/signal/filtering/wiener.cpp
// CPU kernel for Wiener filter using local statistics.
// For each element: compute local mean and variance in a neighborhood,
// then apply y[i] = mean + max(0, var - noise) / max(var, eps) * (x[i] - mean)
// Simplified form: y[i] = x[i] - noise * (x[i] - mean) / max(var, eps)
// inputs[0]: data array (F64 or F32)
// inputs[1]: kernel_size (I32, 1-element, neighborhood half-width)
// outputs[0]: filtered array
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status signal_wiener_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *ks_arr = (InsightArray *)inputs[1];
  InsightArray *y_arr = (InsightArray *)outputs[0];

  if (!x_arr || !ks_arr || !y_arr) {
    cpu_set_last_error("signal_wiener: null array pointer");
    return C_FAILED;
  }

  if (!x_arr->data || !ks_arr->data || !y_arr->data) {
    cpu_set_last_error("signal_wiener: null data pointer");
    return C_FAILED;
  }

  int32_t kernel_size = *(int32_t *)ks_arr->data;
  if (kernel_size < 1)
    kernel_size = 1;

  int64_t n = x_arr->numel;
  double noise = 0.0; // estimated noise variance (use 0 for pure local stats)

  switch (x_arr->dtype) {
  case INSIGHT_DTYPE_F64: {
    const double *x = (const double *)x_arr->data;
    double *y = (double *)y_arr->data;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        int64_t lo = std::max((int64_t)0, i - (int64_t)kernel_size);
        int64_t hi = std::min(n - 1, i + (int64_t)kernel_size);
        int64_t count = hi - lo + 1;
        double sum = 0.0, sum2 = 0.0;
        for (int64_t j = lo; j <= hi; ++j) {
          sum += x[j];
          sum2 += x[j] * x[j];
        }
        double mean = sum / count;
        double var = sum2 / count - mean * mean;
        if (var < 1e-15)
          var = 1e-15;
        y[i] = x[i] - noise * (x[i] - mean) / var;
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        int64_t lo = std::max((int64_t)0, i - (int64_t)kernel_size);
        int64_t hi = std::min(n - 1, i + (int64_t)kernel_size);
        int64_t count = hi - lo + 1;
        double sum = 0.0, sum2 = 0.0;
        for (int64_t j = lo; j <= hi; ++j) {
          sum += x[j];
          sum2 += x[j] * x[j];
        }
        double mean = sum / count;
        double var = sum2 / count - mean * mean;
        if (var < 1e-15)
          var = 1e-15;
        y[i] = x[i] - noise * (x[i] - mean) / var;
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_F32: {
    const float *x = (const float *)x_arr->data;
    float *y = (float *)y_arr->data;
    float noisef = (float)noise;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        int64_t lo = std::max((int64_t)0, i - (int64_t)kernel_size);
        int64_t hi = std::min(n - 1, i + (int64_t)kernel_size);
        int64_t count = hi - lo + 1;
        float sum = 0.0f, sum2 = 0.0f;
        for (int64_t j = lo; j <= hi; ++j) {
          sum += x[j];
          sum2 += x[j] * x[j];
        }
        float mean = sum / count;
        float var = sum2 / count - mean * mean;
        if (var < 1e-15f)
          var = 1e-15f;
        y[i] = x[i] - noisef * (x[i] - mean) / var;
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        int64_t lo = std::max((int64_t)0, i - (int64_t)kernel_size);
        int64_t hi = std::min(n - 1, i + (int64_t)kernel_size);
        int64_t count = hi - lo + 1;
        float sum = 0.0f, sum2 = 0.0f;
        for (int64_t j = lo; j <= hi; ++j) {
          sum += x[j];
          sum2 += x[j] * x[j];
        }
        float mean = sum / count;
        float var = sum2 / count - mean * mean;
        if (var < 1e-15f)
          var = 1e-15f;
        y[i] = x[i] - noisef * (x[i] - mean) / var;
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  default:
    cpu_set_last_error("signal_wiener: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_wiener, INSIGHT_DTYPE_F64, signal_wiener_kernel_cpu);
REGISTER_CPU_KERNEL(signal_wiener, INSIGHT_DTYPE_F32, signal_wiener_kernel_cpu);
