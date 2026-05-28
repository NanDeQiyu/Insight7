// backends/cpu/kernels/elementwise/maximum.cpp
/**
 * @file maximum.cpp
 * @brief CPU kernel for Maximum operation.
 *
 * Computes elementwise ((a > b) ? a : b) of two arrays with stride support.
 * Supports all numeric data types including complex numbers.
 *
 * @param inputs  [0] = InsightArray* left operand
 *                [1] = InsightArray* right operand
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <algorithm>

template <typename T>
static inline std::complex<T> max_complex(std::complex<T> a,
                                          std::complex<T> b) {
  return (a.real() > b.real()) ? a : b;
}

#ifdef __cplusplus
extern "C" {
#endif

#ifdef max
#undef max
#endif

C_Status maximum_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *a = (InsightArray *)inputs[0];
  InsightArray *b = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a || !b || !out) {
    cpu_set_last_error("maximum: null array pointer");
    return C_FAILED;
  }

  // Check that shapes are compatible (numel must match)
  if (a->numel != out->numel || b->numel != out->numel) {
    cpu_set_last_error("maximum: shape mismatch");
    return C_FAILED;
  }

  switch (a->dtype) {
  case INSIGHT_DTYPE_BOOL:
    ELEMENTWISE_KERNEL_LOOP_Func(bool, std::max);
    break;
  case INSIGHT_DTYPE_U8:
    ELEMENTWISE_KERNEL_LOOP_Func(uint8_t, std::max);
    break;
  case INSIGHT_DTYPE_I8:
    ELEMENTWISE_KERNEL_LOOP_Func(int8_t, std::max);
    break;
  case INSIGHT_DTYPE_I16:
    ELEMENTWISE_KERNEL_LOOP_Func(int16_t, std::max);
    break;
  case INSIGHT_DTYPE_I32:
    ELEMENTWISE_KERNEL_LOOP_Func(int32_t, std::max);
    break;
  case INSIGHT_DTYPE_I64:
    ELEMENTWISE_KERNEL_LOOP_Func(int64_t, std::max);
    break;
  case INSIGHT_DTYPE_U16:
    ELEMENTWISE_KERNEL_LOOP_Func(uint16_t, std::max);
    break;
  case INSIGHT_DTYPE_U32:
    ELEMENTWISE_KERNEL_LOOP_Func(uint32_t, std::max);
    break;
  case INSIGHT_DTYPE_U64:
    ELEMENTWISE_KERNEL_LOOP_Func(uint64_t, std::max);
    break;
  case INSIGHT_DTYPE_F32:
    ELEMENTWISE_KERNEL_LOOP_Func(float, std::max);
    break;
  case INSIGHT_DTYPE_F64:
    ELEMENTWISE_KERNEL_LOOP_Func(double, std::max);
    break;
  case INSIGHT_DTYPE_C32:
    ELEMENTWISE_KERNEL_LOOP_Func(std::complex<float>, max_complex);
    break;
  case INSIGHT_DTYPE_C64:
    ELEMENTWISE_KERNEL_LOOP_Func(std::complex<double>, max_complex);
    break;
  default:
    cpu_set_last_error("maximum: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// Register for all supported types
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_BOOL, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_U8, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_I8, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_I16, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_I32, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_I64, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_U16, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_U32, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_U64, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_F16, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_BF16, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_F32, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_F64, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_C32, maximum_kernel_cpu);
REGISTER_CPU_KERNEL(maximum, INSIGHT_DTYPE_C64, maximum_kernel_cpu);
