// backends/cpu/kernels/unary/trunc.cpp
#include "common.h"
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

C_Status trunc_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("trunc: null array pointer");
    return C_FAILED;
  }

  if (x->numel != out->numel) {
    cpu_set_last_error("trunc: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32:
    UNARY_KERNEL_LOOP(float, [](float v) { return std::trunc(v); });
    break;
  case INSIGHT_DTYPE_F64:
    UNARY_KERNEL_LOOP(double, [](double v) { return std::trunc(v); });
    break;
  case INSIGHT_DTYPE_F16:
    UNARY_HALF_LOOP(uint16_t, insight::f16_to_f32, insight::f32_to_f16,
                    [](float v) { return std::trunc(v); });
    break;
  case INSIGHT_DTYPE_BF16:
    UNARY_HALF_LOOP(uint16_t, insight::bf16_to_f32, insight::f32_to_bf16,
                    [](float v) { return std::trunc(v); });
    break;

  default:
    cpu_set_last_error("trunc: only float32 and float64 supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(trunc, INSIGHT_DTYPE_F32, trunc_kernel_cpu);
REGISTER_CPU_KERNEL(trunc, INSIGHT_DTYPE_F64, trunc_kernel_cpu);
REGISTER_CPU_KERNEL(trunc, INSIGHT_DTYPE_F16, trunc_kernel_cpu);
REGISTER_CPU_KERNEL(trunc, INSIGHT_DTYPE_BF16, trunc_kernel_cpu);
