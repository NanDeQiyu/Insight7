// backends/cpu/kernels/signal/windows/blackman.cpp
// CPU kernel for Blackman window generation.
// w[n] = 0.42 - 0.5*cos(2*pi*n/(M-1)) + 0.08*cos(4*pi*n/(M-1))
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status blackman_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("blackman: null output pointer");
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
      double angle = scale * i;
      data[i] = 0.42 - 0.5 * cos(angle) + 0.08 * cos(2.0 * angle);
    }
  } else {
    cpu_set_last_error("blackman: only F64 dtype supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(blackman, INSIGHT_DTYPE_F64, blackman_kernel_cpu);
