// backends/cpu/kernels/signal/windows/cosine.cpp
// CPU kernel for cosine window generation.
// w[n] = sin(pi*(n+0.5)/M)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status cosine_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("cosine: null output pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    double scale = M_PI / M;
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
    for (int64_t i = 0; i < M; ++i) {
      data[i] = sin(scale * (i + 0.5));
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    float scale = (float)M_PI / M;
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
    for (int64_t i = 0; i < M; ++i) {
      data[i] = sinf(scale * (i + 0.5f));
    }
  } else {
    cpu_set_last_error("cosine: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(cosine, INSIGHT_DTYPE_F64, cosine_kernel_cpu);
REGISTER_CPU_KERNEL(cosine, INSIGHT_DTYPE_F32, cosine_kernel_cpu);
