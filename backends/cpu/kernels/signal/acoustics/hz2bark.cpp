// backends/cpu/kernels/signal/acoustics/hz2bark.cpp
// CPU kernel for Hz to Bark conversion.
// bark = 13 * atan(0.00076 * hz) + 3.5 * atan((hz/7500)^2)
// inputs[0]: hz array (F64)
// outputs[0]: bark array (F64)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status signal_hz2bark_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *hz_arr = (InsightArray *)inputs[0];
  InsightArray *bark_arr = (InsightArray *)outputs[0];

  if (!hz_arr || !bark_arr) {
    cpu_set_last_error("signal_hz2bark: null array pointer");
    return C_FAILED;
  }

  if (!hz_arr->data || !bark_arr->data) {
    cpu_set_last_error("signal_hz2bark: null data pointer");
    return C_FAILED;
  }

  int64_t n = hz_arr->numel;

  switch (hz_arr->dtype) {
  case INSIGHT_DTYPE_F64: {
    const double *hz = (const double *)hz_arr->data;
    double *bark = (double *)bark_arr->data;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        double ratio = hz[i] / 7500.0;
        bark[i] =
            13.0 * std::atan(0.00076 * hz[i]) + 3.5 * std::atan(ratio * ratio);
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        double ratio = hz[i] / 7500.0;
        bark[i] =
            13.0 * std::atan(0.00076 * hz[i]) + 3.5 * std::atan(ratio * ratio);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_F32: {
    const float *hz = (const float *)hz_arr->data;
    float *bark = (float *)bark_arr->data;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        float ratio = hz[i] / 7500.0f;
        bark[i] = 13.0f * std::atan(0.00076f * hz[i]) +
                  3.5f * std::atan(ratio * ratio);
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        float ratio = hz[i] / 7500.0f;
        bark[i] = 13.0f * std::atan(0.00076f * hz[i]) +
                  3.5f * std::atan(ratio * ratio);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  default:
    cpu_set_last_error("signal_hz2bark: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_hz2bark, INSIGHT_DTYPE_F64,
                    signal_hz2bark_kernel_cpu);
REGISTER_CPU_KERNEL(signal_hz2bark, INSIGHT_DTYPE_F32,
                    signal_hz2bark_kernel_cpu);
