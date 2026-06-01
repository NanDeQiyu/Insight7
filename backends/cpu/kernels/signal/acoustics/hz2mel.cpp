// backends/cpu/kernels/signal/acoustics/hz2mel.cpp
// CPU kernel for Hz to Mel conversion.
// mel = 2595 * log10(1 + hz/700)
// inputs[0]: hz array (F64)
// outputs[0]: mel array (F64)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status signal_hz2mel_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *hz_arr = (InsightArray *)inputs[0];
  InsightArray *mel_arr = (InsightArray *)outputs[0];

  if (!hz_arr || !mel_arr) {
    cpu_set_last_error("signal_hz2mel: null array pointer");
    return C_FAILED;
  }

  if (!hz_arr->data || !mel_arr->data) {
    cpu_set_last_error("signal_hz2mel: null data pointer");
    return C_FAILED;
  }

  int64_t n = hz_arr->numel;

  switch (hz_arr->dtype) {
  case INSIGHT_DTYPE_F64: {
    const double *hz = (const double *)hz_arr->data;
    double *mel = (double *)mel_arr->data;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        mel[i] = 2595.0 * std::log10(1.0 + hz[i] / 700.0);
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        mel[i] = 2595.0 * std::log10(1.0 + hz[i] / 700.0);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_F32: {
    const float *hz = (const float *)hz_arr->data;
    float *mel = (float *)mel_arr->data;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        mel[i] = 2595.0f * std::log10(1.0f + hz[i] / 700.0f);
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        mel[i] = 2595.0f * std::log10(1.0f + hz[i] / 700.0f);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  default:
    cpu_set_last_error("signal_hz2mel: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_hz2mel, INSIGHT_DTYPE_F64, signal_hz2mel_kernel_cpu);
REGISTER_CPU_KERNEL(signal_hz2mel, INSIGHT_DTYPE_F32, signal_hz2mel_kernel_cpu);
