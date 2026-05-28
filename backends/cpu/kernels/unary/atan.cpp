// backends/cpu/kernels/unary/atan.cpp
#include "common.h"
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

C_Status atan_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("atan: null array pointer");
    return C_FAILED;
  }

  if (x->numel != out->numel) {
    cpu_set_last_error("atan: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32:
    UNARY_KERNEL_LOOP(float, [](float v) { return std::atan(v); });
    break;
  case INSIGHT_DTYPE_F64:
    UNARY_KERNEL_LOOP(double, [](double v) { return std::atan(v); });
    break;
  default:
    cpu_set_last_error("atan: only float32 and float64 supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(atan, INSIGHT_DTYPE_F32, atan_kernel_cpu);
REGISTER_CPU_KERNEL(atan, INSIGHT_DTYPE_F64, atan_kernel_cpu);
