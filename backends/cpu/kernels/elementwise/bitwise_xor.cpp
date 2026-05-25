// backends/cpu/kernels/elementwise/bitwise_xor.cpp
/**
 * @file bitwise_xor.cpp
 * @brief CPU kernel for Bitwise XOR operation.
 *
 * Computes elementwise ^ of two arrays with stride support.
 * Supports all numeric data types including complex numbers.
 *
 * @param inputs  [0] = InsightArray* left operand
 *                [1] = InsightArray* right operand
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status bitwise_xor_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *a = (InsightArray *)inputs[0];
  InsightArray *b = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a || !b || !out) {
    cpu_set_last_error("bitwise_xor: null array pointer");
    return C_FAILED;
  }

  // Check that shapes are compatible (numel must match)
  if (a->numel != out->numel || b->numel != out->numel) {
    cpu_set_last_error("bitwise_xor: shape mismatch");
    return C_FAILED;
  }

  switch (a->dtype) {
  case INSIGHT_DTYPE_BOOL:
    ELEMENTWISE_KERNEL_LOOP(bool, ^);
    break;
  case INSIGHT_DTYPE_U8:
    ELEMENTWISE_KERNEL_LOOP(uint8_t, ^);
    break;
  case INSIGHT_DTYPE_I8:
    ELEMENTWISE_KERNEL_LOOP(int8_t, ^);
    break;
  case INSIGHT_DTYPE_I16:
    ELEMENTWISE_KERNEL_LOOP(int16_t, ^);
    break;
  case INSIGHT_DTYPE_I32:
    ELEMENTWISE_KERNEL_LOOP(int32_t, ^);
    break;
  case INSIGHT_DTYPE_I64:
    ELEMENTWISE_KERNEL_LOOP(int64_t, ^);
    break;
  case INSIGHT_DTYPE_U16:
    ELEMENTWISE_KERNEL_LOOP(uint16_t, ^);
    break;
  case INSIGHT_DTYPE_U32:
    ELEMENTWISE_KERNEL_LOOP(uint32_t, ^);
    break;
  case INSIGHT_DTYPE_U64:
    ELEMENTWISE_KERNEL_LOOP(uint64_t, ^);
    break;
  default:
    cpu_set_last_error("bitwise_xor: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// Register for all supported types
REGISTER_CPU_KERNEL(bitwise_xor, INSIGHT_DTYPE_BOOL, bitwise_xor_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_xor, INSIGHT_DTYPE_U8, bitwise_xor_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_xor, INSIGHT_DTYPE_I8, bitwise_xor_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_xor, INSIGHT_DTYPE_I16, bitwise_xor_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_xor, INSIGHT_DTYPE_I32, bitwise_xor_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_xor, INSIGHT_DTYPE_I64, bitwise_xor_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_xor, INSIGHT_DTYPE_U16, bitwise_xor_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_xor, INSIGHT_DTYPE_U32, bitwise_xor_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_xor, INSIGHT_DTYPE_U64, bitwise_xor_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_xor, INSIGHT_DTYPE_F16, bitwise_xor_kernel_cpu);
REGISTER_CPU_KERNEL(bitwise_xor, INSIGHT_DTYPE_BF16, bitwise_xor_kernel_cpu);
