// backends/cpu/kernels/reduction/count_nonzero.cpp
/**
 * @file count_nonzero.cpp
 * @brief CPU kernel for count_nonzero reduction.
 *
 * Counts number of non-zero elements along reduction axis.
 * Output type is int64.
 *
 * @param inputs  [0] = InsightArray* output (shape [batch_size], dtype int64)
 *                [1] = InsightArray* prepared (shape [batch_size, reduce_size])
 *                [2] = int64_t* batch_size
 *                [3] = int64_t* reduce_size
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status count_nonzero_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *prepared = (InsightArray *)inputs[1];

  if (!out || !prepared) {
    cpu_set_last_error("count_nonzero: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2] || !inputs[3]) {
    cpu_set_last_error("count_nonzero: batch_size or reduce_size is null");
    return C_FAILED;
  }

  switch (prepared->dtype) {
  case INSIGHT_DTYPE_U8:
    REDUCE_COUNT_NONZERO_LOOP(uint8_t);
    break;
  case INSIGHT_DTYPE_I8:
    REDUCE_COUNT_NONZERO_LOOP(int8_t);
    break;
  case INSIGHT_DTYPE_I16:
    REDUCE_COUNT_NONZERO_LOOP(int16_t);
    break;
  case INSIGHT_DTYPE_I32:
    REDUCE_COUNT_NONZERO_LOOP(int32_t);
    break;
  case INSIGHT_DTYPE_I64:
    REDUCE_COUNT_NONZERO_LOOP(int64_t);
    break;
  case INSIGHT_DTYPE_U16:
    REDUCE_COUNT_NONZERO_LOOP(uint16_t);
    break;
  case INSIGHT_DTYPE_U32:
    REDUCE_COUNT_NONZERO_LOOP(uint32_t);
    break;
  case INSIGHT_DTYPE_U64:
    REDUCE_COUNT_NONZERO_LOOP(uint64_t);
    break;
  case INSIGHT_DTYPE_F32:
    REDUCE_COUNT_NONZERO_LOOP(float);
    break;
  case INSIGHT_DTYPE_F64:
    REDUCE_COUNT_NONZERO_LOOP(double);
    break;
  case INSIGHT_DTYPE_BOOL:
    REDUCE_COUNT_NONZERO_LOOP(bool);
    break;
  default:
    cpu_set_last_error("count_nonzero: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(count_nonzero, INSIGHT_DTYPE_U8, count_nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(count_nonzero, INSIGHT_DTYPE_I8, count_nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(count_nonzero, INSIGHT_DTYPE_I16, count_nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(count_nonzero, INSIGHT_DTYPE_I32, count_nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(count_nonzero, INSIGHT_DTYPE_I64, count_nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(count_nonzero, INSIGHT_DTYPE_U16, count_nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(count_nonzero, INSIGHT_DTYPE_U32, count_nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(count_nonzero, INSIGHT_DTYPE_U64, count_nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(count_nonzero, INSIGHT_DTYPE_F32, count_nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(count_nonzero, INSIGHT_DTYPE_F64, count_nonzero_kernel_cpu);
REGISTER_CPU_KERNEL(count_nonzero, INSIGHT_DTYPE_BOOL,
                    count_nonzero_kernel_cpu);
