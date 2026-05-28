// backends/cpu/kernels/indexing/where.cpp
/**
 * @file where.cpp
 * @brief CPU kernel for where operation.
 *
 * Returns elements from x where condition is true, from y where false.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output
 *   inputs[1] = InsightArray* condition (bool)
 *   inputs[2] = InsightArray* x
 *   inputs[3] = InsightArray* y
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status where_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *cond = (InsightArray *)inputs[1];
  InsightArray *x = (InsightArray *)inputs[2];
  InsightArray *y = (InsightArray *)inputs[3];

  if (!out || !cond || !x || !y) {
    cpu_set_last_error("where: null array pointer");
    return C_FAILED;
  }

  int64_t n = out->numel;
  const bool *cond_data = (const bool *)cond->data;

  // Assuming all inputs are contiguous and have same shape
  switch (x->dtype) {
  case INSIGHT_DTYPE_F32: {
    float *dst = (float *)out->data;
    const float *x_data = (const float *)x->data;
    const float *y_data = (const float *)y->data;
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = cond_data[i] ? x_data[i] : y_data[i];
    }
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *dst = (double *)out->data;
    const double *x_data = (const double *)x->data;
    const double *y_data = (const double *)y->data;
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = cond_data[i] ? x_data[i] : y_data[i];
    }
    break;
  }
  case INSIGHT_DTYPE_I32: {
    int32_t *dst = (int32_t *)out->data;
    const int32_t *x_data = (const int32_t *)x->data;
    const int32_t *y_data = (const int32_t *)y->data;
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = cond_data[i] ? x_data[i] : y_data[i];
    }
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t *dst = (int64_t *)out->data;
    const int64_t *x_data = (const int64_t *)x->data;
    const int64_t *y_data = (const int64_t *)y->data;
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = cond_data[i] ? x_data[i] : y_data[i];
    }
    break;
  }
  case INSIGHT_DTYPE_U8: {
    uint8_t *dst = (uint8_t *)out->data;
    const uint8_t *x_data = (const uint8_t *)x->data;
    const uint8_t *y_data = (const uint8_t *)y->data;
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = cond_data[i] ? x_data[i] : y_data[i];
    }
    break;
  }
  case INSIGHT_DTYPE_BOOL: {
    bool *dst = (bool *)out->data;
    const bool *x_data = (const bool *)x->data;
    const bool *y_data = (const bool *)y->data;
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = cond_data[i] ? x_data[i] : y_data[i];
    }
    break;
  }
  default:
    cpu_set_last_error("where: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(where, INSIGHT_DTYPE_F32, where_kernel_cpu);
REGISTER_CPU_KERNEL(where, INSIGHT_DTYPE_F64, where_kernel_cpu);
REGISTER_CPU_KERNEL(where, INSIGHT_DTYPE_I32, where_kernel_cpu);
REGISTER_CPU_KERNEL(where, INSIGHT_DTYPE_I64, where_kernel_cpu);
REGISTER_CPU_KERNEL(where, INSIGHT_DTYPE_U8, where_kernel_cpu);
REGISTER_CPU_KERNEL(where, INSIGHT_DTYPE_BOOL, where_kernel_cpu);
