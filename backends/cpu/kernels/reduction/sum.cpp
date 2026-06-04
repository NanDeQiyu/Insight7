// backends/cpu/kernels/reduction/sum.cpp
/**
 * @file sum.cpp
 * @brief CPU kernel for Sum reduction.
 *
 * Performs reduction along the last dimension.
 * Input is guaranteed to be contiguous with shape [batch_size, reduce_size].
 *
 * @param inputs  [0] = InsightArray* output (shape [batch_size])
 *                [1] = InsightArray* prepared (shape [batch_size, reduce_size])
 *                [2] = int64_t* batch_size
 *                [3] = int64_t* reduce_size
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

C_Status sum_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *prepared = (InsightArray *)inputs[1];

  if (!out || !prepared) {
    cpu_set_last_error("sum: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2] || !inputs[3]) {
    cpu_set_last_error("sum: batch_size or reduce_size is null");
    return C_FAILED;
  }

  switch (out->dtype) {
  case INSIGHT_DTYPE_U8:
    REDUCE_SUM_LOOP(uint8_t);
    break;
  case INSIGHT_DTYPE_I8:
    REDUCE_SUM_LOOP(int8_t);
    break;
  case INSIGHT_DTYPE_I16:
    REDUCE_SUM_LOOP(int16_t);
    break;
  case INSIGHT_DTYPE_I32:
    REDUCE_SUM_LOOP(int32_t);
    break;
  case INSIGHT_DTYPE_I64:
    REDUCE_SUM_LOOP(int64_t);
    break;
  case INSIGHT_DTYPE_U16:
    REDUCE_SUM_LOOP(uint16_t);
    break;
  case INSIGHT_DTYPE_U32:
    REDUCE_SUM_LOOP(uint32_t);
    break;
  case INSIGHT_DTYPE_U64:
    REDUCE_SUM_LOOP(uint64_t);
    break;
  case INSIGHT_DTYPE_F32:
    REDUCE_SUM_LOOP(float);
    break;
  case INSIGHT_DTYPE_F64:
    REDUCE_SUM_LOOP(double);
    break;
  case INSIGHT_DTYPE_F16:
    REDUCE_HALF_SUM_LOOP(uint16_t, insight::f16_to_f32, insight::f32_to_f16);
    break;
  case INSIGHT_DTYPE_BF16:
    REDUCE_HALF_SUM_LOOP(uint16_t, insight::bf16_to_f32, insight::f32_to_bf16);
    break;
  case INSIGHT_DTYPE_C32:
    REDUCE_SUM_LOOP(std::complex<float>);
    break;
  case INSIGHT_DTYPE_C64:
    REDUCE_SUM_LOOP(std::complex<double>);
    break;

  default:
    cpu_set_last_error("sum: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// Register for all supported types
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_U8, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_I8, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_I16, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_I32, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_I64, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_U16, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_U32, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_U64, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_F32, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_F64, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_F16, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_BF16, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_C32, sum_kernel_cpu);
REGISTER_CPU_KERNEL(sum, INSIGHT_DTYPE_C64, sum_kernel_cpu);
