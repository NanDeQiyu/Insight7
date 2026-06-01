// backends/cpu/kernels/reduction/argmin.cpp
/**
 * @file argmin.cpp
 * @brief CPU kernel for ArgMin reduction.
 *
 * Returns index of maximum/minimum value along reduction axis.
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

C_Status argmin_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *prepared = (InsightArray *)inputs[1];

  if (!out || !prepared) {
    cpu_set_last_error("argmin: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2] || !inputs[3]) {
    cpu_set_last_error("argmin: batch_size or reduce_size is null");
    return C_FAILED;
  }

  switch (prepared->dtype) {
  case INSIGHT_DTYPE_U8:
    REDUCE_ARGMIN_LOOP(uint8_t);
    break;
  case INSIGHT_DTYPE_I8:
    REDUCE_ARGMIN_LOOP(int8_t);
    break;
  case INSIGHT_DTYPE_I16:
    REDUCE_ARGMIN_LOOP(int16_t);
    break;
  case INSIGHT_DTYPE_I32:
    REDUCE_ARGMIN_LOOP(int32_t);
    break;
  case INSIGHT_DTYPE_I64:
    REDUCE_ARGMIN_LOOP(int64_t);
    break;
  case INSIGHT_DTYPE_U16:
    REDUCE_ARGMIN_LOOP(uint16_t);
    break;
  case INSIGHT_DTYPE_U32:
    REDUCE_ARGMIN_LOOP(uint32_t);
    break;
  case INSIGHT_DTYPE_U64:
    REDUCE_ARGMIN_LOOP(uint64_t);
    break;
  case INSIGHT_DTYPE_F32:
    REDUCE_ARGMIN_LOOP(float);
    break;
  case INSIGHT_DTYPE_F64:
    REDUCE_ARGMIN_LOOP(double);
    break;
  case INSIGHT_DTYPE_F16: {
    int64_t *dst = (int64_t *)out->data;
    const uint16_t *src = (const uint16_t *)prepared->data;
    int64_t batch_size = *(int64_t *)inputs[2];
    int64_t reduce_size = *(int64_t *)inputs[3];
    _Pragma("omp parallel for") for (int64_t i = 0; i < batch_size; ++i) {
      int64_t min_idx = 0;
      float min_val = insight::f16_to_f32(src[i * reduce_size]);
      for (int64_t j = 1; j < reduce_size; ++j) {
        float val = insight::f16_to_f32(src[i * reduce_size + j]);
        if (val < min_val) {
          min_val = val;
          min_idx = j;
        }
      }
      dst[i] = min_idx;
    }
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    int64_t *dst = (int64_t *)out->data;
    const uint16_t *src = (const uint16_t *)prepared->data;
    int64_t batch_size = *(int64_t *)inputs[2];
    int64_t reduce_size = *(int64_t *)inputs[3];
    _Pragma("omp parallel for") for (int64_t i = 0; i < batch_size; ++i) {
      int64_t min_idx = 0;
      float min_val = insight::bf16_to_f32(src[i * reduce_size]);
      for (int64_t j = 1; j < reduce_size; ++j) {
        float val = insight::bf16_to_f32(src[i * reduce_size + j]);
        if (val < min_val) {
          min_val = val;
          min_idx = j;
        }
      }
      dst[i] = min_idx;
    }
    break;
  }
  default:
    cpu_set_last_error("argmin: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_U8, argmin_kernel_cpu);
REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_I8, argmin_kernel_cpu);
REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_I16, argmin_kernel_cpu);
REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_I32, argmin_kernel_cpu);
REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_I64, argmin_kernel_cpu);
REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_U16, argmin_kernel_cpu);
REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_U32, argmin_kernel_cpu);
REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_U64, argmin_kernel_cpu);
REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_F32, argmin_kernel_cpu);
REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_F64, argmin_kernel_cpu);
REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_F16, argmin_kernel_cpu);
REGISTER_CPU_KERNEL(argmin, INSIGHT_DTYPE_BF16, argmin_kernel_cpu);
