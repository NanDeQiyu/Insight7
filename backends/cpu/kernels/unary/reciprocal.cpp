// backends/cpu/kernels/unary/reciprocal.cpp
#include "common.h"
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

C_Status reciprocal_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("reciprocal: null array pointer");
    return C_FAILED;
  }

  if (x->numel != out->numel) {
    cpu_set_last_error("reciprocal: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32:
    UNARY_KERNEL_LOOP(float, [](float v) { return 1.0f / v; });
    break;
  case INSIGHT_DTYPE_F64:
    UNARY_KERNEL_LOOP(double, [](double v) { return 1.0 / v; });
    break;
  case INSIGHT_DTYPE_C32:
    UNARY_KERNEL_LOOP(std::complex<float>,
                      [](std::complex<float> v) { return 1.0f / v; });
    break;
  case INSIGHT_DTYPE_C64:
    UNARY_KERNEL_LOOP(std::complex<double>,
                      [](std::complex<double> v) { return 1.0 / v; });
    break;
  default:
    cpu_set_last_error("reciprocal: only float and complex types supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(reciprocal, INSIGHT_DTYPE_F32, reciprocal_kernel_cpu);
REGISTER_CPU_KERNEL(reciprocal, INSIGHT_DTYPE_F64, reciprocal_kernel_cpu);
REGISTER_CPU_KERNEL(reciprocal, INSIGHT_DTYPE_C32, reciprocal_kernel_cpu);
REGISTER_CPU_KERNEL(reciprocal, INSIGHT_DTYPE_C64, reciprocal_kernel_cpu);
