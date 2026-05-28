// backends/cpu/kernels/random/chisquare.cpp
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status chisquare_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("chisquare: output array is null");
    return C_FAILED;
  }

  double df = *(double *)inputs[1];

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    RANDOM_FILL_LOOP(float, std::chi_squared_distribution<float>, ((float)df));
    break;
  case INSIGHT_DTYPE_F64:
    RANDOM_FILL_LOOP(double, std::chi_squared_distribution<double>, (df));
    break;
  default:
    cpu_set_last_error("chisquare: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(chisquare, INSIGHT_DTYPE_F32, chisquare_kernel_cpu);
REGISTER_CPU_KERNEL(chisquare, INSIGHT_DTYPE_F64, chisquare_kernel_cpu);
