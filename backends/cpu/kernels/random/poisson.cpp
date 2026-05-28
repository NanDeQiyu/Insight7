// backends/cpu/kernels/random/poisson.cpp
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status poisson_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("poisson: output array is null");
    return C_FAILED;
  }

  double lam = *(double *)inputs[1];

  switch (out->dtype) {
  case INSIGHT_DTYPE_I32:
    RANDOM_FILL_LOOP(int32_t, std::poisson_distribution<int32_t>, (lam));
    break;
  case INSIGHT_DTYPE_I64:
    RANDOM_FILL_LOOP(int64_t, std::poisson_distribution<int64_t>, (lam));
    break;
  default:
    cpu_set_last_error("poisson: unsupported dtype (need int32/int64)");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(poisson, INSIGHT_DTYPE_I32, poisson_kernel_cpu);
REGISTER_CPU_KERNEL(poisson, INSIGHT_DTYPE_I64, poisson_kernel_cpu);
