// backends/cpu/kernels/signal/windows/bartlett.cpp
// CPU kernel for Bartlett window generation.
// w[n] = 1 - |2*n/(M-1) - 1|
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status bartlett_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("bartlett: null output pointer");
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
    double denom = (double)(M - 1);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
    for (int64_t i = 0; i < M; ++i) {
      data[i] = 1.0 - fabs(2.0 * i / denom - 1.0);
    }
  } else {
    cpu_set_last_error("bartlett: only F64 dtype supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(bartlett, INSIGHT_DTYPE_F64, bartlett_kernel_cpu);
