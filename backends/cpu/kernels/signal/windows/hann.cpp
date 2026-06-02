// backends/cpu/kernels/signal/windows/hann.cpp
// CPU kernel for Hann window generation.
// w[n] = 0.5 * (1 - cos(2*pi*n/(M-1)))
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status hann_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("hann: null output pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    if (M == 1) {
      data[0] = 1.0;
      return C_SUCCESS;
    }
    double scale = 2.0 * M_PI / (M - 1);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
    for (int64_t i = 0; i < M; ++i) {
      data[i] = 0.5 * (1.0 - cos(scale * i));
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    if (M == 1) {
      data[0] = 1.0f;
      return C_SUCCESS;
    }
    float scale = 2.0f * (float)M_PI / (M - 1);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
    for (int64_t i = 0; i < M; ++i) {
      data[i] = 0.5f * (1.0f - cosf(scale * i));
    }
  } else {
    cpu_set_last_error("hann: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(hann, INSIGHT_DTYPE_F64, hann_kernel_cpu);
REGISTER_CPU_KERNEL(hann, INSIGHT_DTYPE_F32, hann_kernel_cpu);
