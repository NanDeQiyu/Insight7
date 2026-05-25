// backends/cpu/kernels/elementwise/greater_equal.cpp
/**
 * @file greater_equal.cpp
 * @brief CPU kernel for Greater Than or Equal operation.
 *
 * Computes elementwise >= of two arrays with stride support.
 * Supports all numeric data types including complex numbers.
 *
 * @param inputs  [0] = InsightArray* left operand
 *                [1] = InsightArray* right operand
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

static inline bool cpu_complex_float_ge(std::complex<float> a,
                                        std::complex<float> b) {
  return a.real() >= b.real();
}

static inline bool cpu_complex_double_ge(std::complex<double> a,
                                         std::complex<double> b) {
  return a.real() >= b.real();
}

#ifdef __cplusplus
extern "C" {
#endif

C_Status greater_equal_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *a = (InsightArray *)inputs[0];
  InsightArray *b = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a || !b || !out) {
    cpu_set_last_error("greater_equal: null array pointer");
    return C_FAILED;
  }

  // Check that shapes are compatible (numel must match)
  if (a->numel != out->numel || b->numel != out->numel) {
    cpu_set_last_error("greater_equal: shape mismatch");
    return C_FAILED;
  }

  switch (a->dtype) {
  case INSIGHT_DTYPE_BOOL:
    ELEMENTWISE_CMP_LOOP(bool, >=);
    break;
  case INSIGHT_DTYPE_U8:
    ELEMENTWISE_CMP_LOOP(uint8_t, >=);
    break;
  case INSIGHT_DTYPE_I8:
    ELEMENTWISE_CMP_LOOP(int8_t, >=);
    break;
  case INSIGHT_DTYPE_I16:
    ELEMENTWISE_CMP_LOOP(int16_t, >=);
    break;
  case INSIGHT_DTYPE_I32:
    ELEMENTWISE_CMP_LOOP(int32_t, >=);
    break;
  case INSIGHT_DTYPE_I64:
    ELEMENTWISE_CMP_LOOP(int64_t, >=);
    break;
  case INSIGHT_DTYPE_U16:
    ELEMENTWISE_CMP_LOOP(uint16_t, >=);
    break;
  case INSIGHT_DTYPE_U32:
    ELEMENTWISE_CMP_LOOP(uint32_t, >=);
    break;
  case INSIGHT_DTYPE_U64:
    ELEMENTWISE_CMP_LOOP(uint64_t, >=);
    break;
  case INSIGHT_DTYPE_F32:
    ELEMENTWISE_CMP_LOOP(float, >=);
    break;
  case INSIGHT_DTYPE_F64:
    ELEMENTWISE_CMP_LOOP(double, >=);
    break;
  case INSIGHT_DTYPE_C32:
    ELEMENTWISE_CMP_LOOP_Func(std::complex<float>, cpu_complex_float_ge);
    break;
  case INSIGHT_DTYPE_C64:
    ELEMENTWISE_CMP_LOOP_Func(std::complex<double>, cpu_complex_double_ge);
    break;
  default:
    cpu_set_last_error("greater_equal: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// Register for all supported types
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_BOOL,
                    greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_U8, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_I8, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_I16, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_I32, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_I64, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_U16, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_U32, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_U64, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_F16, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_BF16,
                    greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_F32, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_F64, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_C32, greater_equal_kernel_cpu);
REGISTER_CPU_KERNEL(greater_equal, INSIGHT_DTYPE_C64, greater_equal_kernel_cpu);
