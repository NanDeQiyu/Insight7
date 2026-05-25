// backends/cpu/kernels/reduction/nanmax.cpp
/**
 * @file nanmax.cpp
 * @brief CPU kernel for nanmax reduction (maximum ignoring NaN).
 *
 * Finds maximum value along reduction axis, skipping NaN values.
 * If all elements are NaN, returns 0.
 *
 * @param inputs  [0] = InsightArray* output (shape [batch_size])
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

C_Status nanmax_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *prepared = (InsightArray *)inputs[1];

  if (!out || !prepared) {
    cpu_set_last_error("nanmax: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2] || !inputs[3]) {
    cpu_set_last_error("nanmax: batch_size or reduce_size is null");
    return C_FAILED;
  }

  switch (prepared->dtype) {
  case INSIGHT_DTYPE_F32:
    REDUCE_NANMAX_LOOP(float, cpu_is_nan_f32);
    break;
  case INSIGHT_DTYPE_F64:
    REDUCE_NANMAX_LOOP(double, cpu_is_nan_f64);
    break;
  default:
    cpu_set_last_error("nanmax: only float32 and float64 supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(nanmax, INSIGHT_DTYPE_F32, nanmax_kernel_cpu);
REGISTER_CPU_KERNEL(nanmax, INSIGHT_DTYPE_F64, nanmax_kernel_cpu);
