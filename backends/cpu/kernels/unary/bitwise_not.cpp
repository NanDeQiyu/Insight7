// backends/cpu/kernels/unary/bitwise_not.cpp
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status bitwise_not_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("bitwise_not: null array pointer");
    return C_FAILED;
  }

  if (x->numel != out->numel) {
    cpu_set_last_error("bitwise_not: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_BOOL:
    UNARY_KERNEL_LOOP(bool, [](bool v) { return !v; });
    break;
  case INSIGHT_DTYPE_U8:
    UNARY_KERNEL_LOOP(uint8_t, [](uint8_t v) { return ~v; });
    break;
  case INSIGHT_DTYPE_I8:
    UNARY_KERNEL_LOOP(int8_t, [](int8_t v) { return ~v; });
    break;
  case INSIGHT_DTYPE_I16:
    UNARY_KERNEL_LOOP(int16_t, [](int16_t v) { return ~v; });
    break;
  case INSIGHT_DTYPE_I32:
    UNARY_KERNEL_LOOP(int32_t, [](int32_t v) { return ~v; });
    break;
  case INSIGHT_DTYPE_I64:
    UNARY_KERNEL_LOOP(int64_t, [](int64_t v) { return ~v; });
    break;
  case INSIGHT_DTYPE_U16:
    UNARY_KERNEL_LOOP(uint16_t, [](uint16_t v) { return ~v; });
    break;
  case INSIGHT_DTYPE_U32:
    UNARY_KERNEL_LOOP(uint32_t, [](uint32_t v) { return ~v; });
    break;
  case INSIGHT_DTYPE_U64:
    UNARY_KERNEL_LOOP(uint64_t, [](uint64_t v) { return ~v; });
    break;
  default:
    cpu_set_last_error("bitwise_not: only integer and bool types supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(bitwise_not, INSIGHT_DTYPE_BOOL, bitwise_not_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_not, INSIGHT_DTYPE_U8, bitwise_not_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_not, INSIGHT_DTYPE_I8, bitwise_not_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_not, INSIGHT_DTYPE_I16, bitwise_not_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_not, INSIGHT_DTYPE_I32, bitwise_not_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_not, INSIGHT_DTYPE_I64, bitwise_not_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_not, INSIGHT_DTYPE_U16, bitwise_not_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_not, INSIGHT_DTYPE_U32, bitwise_not_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_not, INSIGHT_DTYPE_U64, bitwise_not_kernel_cpu);
