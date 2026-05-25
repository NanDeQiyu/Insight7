// backends/cpu/kernels/unary/negative.cpp
#include "common.h"
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

C_Status negative_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("negative: null array pointer");
    return C_FAILED;
  }

  if (x->numel != out->numel) {
    cpu_set_last_error("negative: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_U8:
  case INSIGHT_DTYPE_U16:
  case INSIGHT_DTYPE_U32:
  case INSIGHT_DTYPE_U64:
    cpu_set_last_error("negative: unsigned integers not supported");
    return C_FAILED;
  case INSIGHT_DTYPE_I8:
    UNARY_KERNEL_LOOP(int8_t, [](int8_t v) { return -v; });
    break;
  case INSIGHT_DTYPE_I16:
    UNARY_KERNEL_LOOP(int16_t, [](int16_t v) { return -v; });
    break;
  case INSIGHT_DTYPE_I32:
    UNARY_KERNEL_LOOP(int32_t, [](int32_t v) { return -v; });
    break;
  case INSIGHT_DTYPE_I64:
    UNARY_KERNEL_LOOP(int64_t, [](int64_t v) { return -v; });
    break;
  case INSIGHT_DTYPE_F32:
    UNARY_KERNEL_LOOP(float, [](float v) { return -v; });
    break;
  case INSIGHT_DTYPE_F64:
    UNARY_KERNEL_LOOP(double, [](double v) { return -v; });
    break;
  case INSIGHT_DTYPE_C32:
    UNARY_KERNEL_LOOP(std::complex<float>,
                      [](std::complex<float> v) { return -v; });
    break;
  case INSIGHT_DTYPE_C64:
    UNARY_KERNEL_LOOP(std::complex<double>,
                      [](std::complex<double> v) { return -v; });
    break;
  default:
    cpu_set_last_error("negative: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(negative, INSIGHT_DTYPE_I8, negative_kernel_cpu);
REGISTER_CPU_KERNEL(negative, INSIGHT_DTYPE_I16, negative_kernel_cpu);
REGISTER_CPU_KERNEL(negative, INSIGHT_DTYPE_I32, negative_kernel_cpu);
REGISTER_CPU_KERNEL(negative, INSIGHT_DTYPE_I64, negative_kernel_cpu);
REGISTER_CPU_KERNEL(negative, INSIGHT_DTYPE_F32, negative_kernel_cpu);
REGISTER_CPU_KERNEL(negative, INSIGHT_DTYPE_F64, negative_kernel_cpu);
REGISTER_CPU_KERNEL(negative, INSIGHT_DTYPE_C32, negative_kernel_cpu);
REGISTER_CPU_KERNEL(negative, INSIGHT_DTYPE_C64, negative_kernel_cpu);
