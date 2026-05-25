// backends/cpu/kernels/unary/rad2deg.cpp
#include "common.h"
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

C_Status rad2deg_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("rad2deg: null array pointer");
    return C_FAILED;
  }

  if (x->numel != out->numel) {
    cpu_set_last_error("rad2deg: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32:
    UNARY_KERNEL_LOOP(float,
                      [](float v) { return v * 180.0f / 3.141592653589793f; });
    break;
  case INSIGHT_DTYPE_F64:
    UNARY_KERNEL_LOOP(double,
                      [](double v) { return v * 180.0 / 3.141592653589793; });
    break;
  default:
    cpu_set_last_error("rad2deg: only float32 and float64 supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(rad2deg, INSIGHT_DTYPE_F32, rad2deg_kernel_cpu);
REGISTER_CPU_KERNEL(rad2deg, INSIGHT_DTYPE_F64, rad2deg_kernel_cpu);