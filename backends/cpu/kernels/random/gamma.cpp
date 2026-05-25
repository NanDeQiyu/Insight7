// backends/cpu/kernels/random/gamma.cpp
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status gamma_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("gamma: output array is null");
    return C_FAILED;
  }

  double shape_param = *(double *)inputs[1];
  double rate = *(double *)inputs[2];

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    RANDOM_FILL_LOOP(float, std::gamma_distribution<float>,
                     (shape_param, 1.0f / rate));
    break;
  case INSIGHT_DTYPE_F64:
    RANDOM_FILL_LOOP(double, std::gamma_distribution<double>,
                     (shape_param, 1.0 / rate));
    break;
  default:
    cpu_set_last_error("gamma: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(gamma, INSIGHT_DTYPE_F32, gamma_kernel_cpu);
REGISTER_CPU_KERNEL(gamma, INSIGHT_DTYPE_F64, gamma_kernel_cpu);