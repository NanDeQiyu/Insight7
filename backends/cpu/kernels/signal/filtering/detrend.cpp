// backends/cpu/kernels/signal/filtering/detrend.cpp
// CPU kernel for linear detrending.
// Fits y = a + b*x via least squares, subtracts trend from data.
// inputs[0]: data array (F64 or F32)
// outputs[0]: detrended array
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status signal_detrend_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *y_arr = (InsightArray *)outputs[0];

  if (!x_arr || !y_arr) {
    cpu_set_last_error("signal_detrend: null array pointer");
    return C_FAILED;
  }

  if (!x_arr->data || !y_arr->data) {
    cpu_set_last_error("signal_detrend: null data pointer");
    return C_FAILED;
  }

  int64_t n = x_arr->numel;
  if (n == 0)
    return C_SUCCESS;

  // Compute sums for least-squares fit: y = a + b*x
  // x indices: 0, 1, ..., n-1
  // sum_x = n*(n-1)/2, sum_x2 = (n-1)*n*(2n-1)/6
  double sum_x = (double)n * (double)(n - 1) * 0.5;
  double sum_x2 = (double)(n - 1) * (double)n * (double)(2 * n - 1) / 6.0;
  double nd = (double)n;
  double denom = nd * sum_x2 - sum_x * sum_x;

  switch (x_arr->dtype) {
  case INSIGHT_DTYPE_F64: {
    const double *x = (const double *)x_arr->data;
    double *y = (double *)y_arr->data;

    // Compute sum_y and sum_xy
    double sum_y = 0.0, sum_xy = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      sum_y += x[i];
      sum_xy += (double)i * x[i];
    }

    double a = 0.0, b = 0.0;
    if (std::abs(denom) > 1e-30) {
      a = (sum_y * sum_x2 - sum_x * sum_xy) / denom;
      b = (nd * sum_xy - sum_x * sum_y) / denom;
    } else {
      a = sum_y / nd;
    }

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        y[i] = x[i] - (a + b * (double)i);
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        y[i] = x[i] - (a + b * (double)i);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_F32: {
    const float *x = (const float *)x_arr->data;
    float *y = (float *)y_arr->data;

    double sum_y = 0.0, sum_xy = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      sum_y += (double)x[i];
      sum_xy += (double)i * (double)x[i];
    }

    double a = 0.0, b = 0.0;
    if (std::abs(denom) > 1e-30) {
      a = (sum_y * sum_x2 - sum_x * sum_xy) / denom;
      b = (nd * sum_xy - sum_x * sum_y) / denom;
    } else {
      a = sum_y / nd;
    }

    float af = (float)a, bf = (float)b;
#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        y[i] = x[i] - (af + bf * (float)i);
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        y[i] = x[i] - (af + bf * (float)i);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  default:
    cpu_set_last_error("signal_detrend: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_detrend, INSIGHT_DTYPE_F64,
                    signal_detrend_kernel_cpu);
REGISTER_CPU_KERNEL(signal_detrend, INSIGHT_DTYPE_F32,
                    signal_detrend_kernel_cpu);
