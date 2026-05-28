// backends/cpu/kernels/random/uniform.cpp
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status uniform_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("uniform: output array is null");
    return C_FAILED;
  }

  double low = *(double *)inputs[1];
  double high = *(double *)inputs[2];

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    RANDOM_FILL_LOOP(float, std::uniform_real_distribution<float>, (low, high));
    break;
  case INSIGHT_DTYPE_F64:
    RANDOM_FILL_LOOP(double, std::uniform_real_distribution<double>,
                     (low, high));
    break;
  default:
    cpu_set_last_error("uniform: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(uniform, INSIGHT_DTYPE_F32, uniform_kernel_cpu);
REGISTER_CPU_KERNEL(uniform, INSIGHT_DTYPE_F64, uniform_kernel_cpu);