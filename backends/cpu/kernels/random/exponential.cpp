// backends/cpu/kernels/random/exponential.cpp
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status exponential_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("exponential: output array is null");
    return C_FAILED;
  }

  double scale = *(double *)inputs[1];

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    RANDOM_FILL_LOOP(float, std::exponential_distribution<float>,
                     (1.0f / scale));
    break;
  case INSIGHT_DTYPE_F64:
    RANDOM_FILL_LOOP(double, std::exponential_distribution<double>,
                     (1.0 / scale));
    break;
  default:
    cpu_set_last_error("exponential: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(exponential, INSIGHT_DTYPE_F32, exponential_kernel_cpu);
REGISTER_CPU_KERNEL(exponential, INSIGHT_DTYPE_F64, exponential_kernel_cpu);