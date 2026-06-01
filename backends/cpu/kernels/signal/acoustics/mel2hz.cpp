// backends/cpu/kernels/signal/acoustics/mel2hz.cpp
// CPU kernel for Mel to Hz conversion.
// hz = 700 * (10^(mel/2595) - 1) for standard
// inputs[0]: mel array (F64)
// outputs[0]: hz array (F64)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status signal_mel2hz_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *mel_arr = (InsightArray *)inputs[0];
  InsightArray *hz_arr = (InsightArray *)outputs[0];

  if (!mel_arr || !hz_arr) {
    cpu_set_last_error("signal_mel2hz: null array pointer");
    return C_FAILED;
  }

  if (!mel_arr->data || !hz_arr->data) {
    cpu_set_last_error("signal_mel2hz: null data pointer");
    return C_FAILED;
  }

  int64_t n = mel_arr->numel;

  switch (mel_arr->dtype) {
  case INSIGHT_DTYPE_F64: {
    const double *mel = (const double *)mel_arr->data;
    double *hz = (double *)hz_arr->data;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        hz[i] = 700.0 * (std::pow(10.0, mel[i] / 2595.0) - 1.0);
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        hz[i] = 700.0 * (std::pow(10.0, mel[i] / 2595.0) - 1.0);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_F32: {
    const float *mel = (const float *)mel_arr->data;
    float *hz = (float *)hz_arr->data;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        hz[i] = 700.0f * (std::pow(10.0f, mel[i] / 2595.0f) - 1.0f);
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        hz[i] = 700.0f * (std::pow(10.0f, mel[i] / 2595.0f) - 1.0f);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  default:
    cpu_set_last_error("signal_mel2hz: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_mel2hz, INSIGHT_DTYPE_F64, signal_mel2hz_kernel_cpu);
REGISTER_CPU_KERNEL(signal_mel2hz, INSIGHT_DTYPE_F32, signal_mel2hz_kernel_cpu);
