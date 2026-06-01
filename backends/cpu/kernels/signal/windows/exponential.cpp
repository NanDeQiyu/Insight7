// backends/cpu/kernels/signal/windows/exponential.cpp
// CPU kernel for exponential window generation.
// w[n] = exp(-|n - center| / tau)
// inputs[0]: tau   (1-element F64 array)
// inputs[1]: center (1-element F64 array)
// outputs[0]: window output
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status signal_exponential_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *tau_arr = (InsightArray *)inputs[0];
  InsightArray *center_arr = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!tau_arr || !center_arr || !out) {
    cpu_set_last_error("exponential: null array pointer");
    return C_FAILED;
  }

  double tau = *(double *)tau_arr->data;
  double center = *(double *)center_arr->data;
  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    if (tau <= 0.0) {
      cpu_set_last_error("exponential: tau must be positive");
      return C_FAILED;
    }
    double inv_tau = 1.0 / tau;
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
    for (int64_t i = 0; i < M; ++i) {
      data[i] = exp(-fabs(i - center) * inv_tau);
    }
  } else {
    cpu_set_last_error("exponential: only F64 dtype supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_exponential, INSIGHT_DTYPE_F64,
                    signal_exponential_kernel_cpu);
