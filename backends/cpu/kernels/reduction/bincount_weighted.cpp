// backends/cpu/kernels/reduction/bincount_weighted.cpp
/**
 * @file bincount_weighted.cpp
 * @brief CPU kernel for weighted bincount.
 *
 * Sums weights for each integer value in the input array.
 * Output array is pre-initialized to zeros.
 *
 * @param inputs  [0] = InsightArray* output (pre-zeroed)
 *                [1] = InsightArray* indices (1D integer array)
 *                [2] = InsightArray* weights (1D array)
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_BINCOUNT_WEIGHTED(ITYPE, WTYPE)                                 \
  do {                                                                         \
    const ITYPE *idx_src = (const ITYPE *)x->data;                             \
    const WTYPE *w_src = (const WTYPE *)weights->data;                         \
    WTYPE *dst = (WTYPE *)out->data;                                           \
    int64_t n = x->numel;                                                      \
    _Pragma("omp parallel for") for (int64_t i = 0; i < n; ++i) {              \
      int64_t idx = idx_src[i];                                                \
      _Pragma("omp atomic") dst[idx] += w_src[i];                              \
    }                                                                          \
  } while (0)

C_Status bincount_weighted_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  InsightArray *weights = (InsightArray *)inputs[2];

  if (!out || !x || !weights) {
    cpu_set_last_error("bincount_weighted: null array pointer");
    return C_FAILED;
  }

  if (x->dtype == INSIGHT_DTYPE_I32) {
    switch (weights->dtype) {
    case INSIGHT_DTYPE_F32:
      DEFINE_BINCOUNT_WEIGHTED(int32_t, float);
      break;
    case INSIGHT_DTYPE_F64:
      DEFINE_BINCOUNT_WEIGHTED(int32_t, double);
      break;
    case INSIGHT_DTYPE_I32:
      DEFINE_BINCOUNT_WEIGHTED(int32_t, int32_t);
      break;
    case INSIGHT_DTYPE_I64:
      DEFINE_BINCOUNT_WEIGHTED(int32_t, int64_t);
      break;
    default:
      cpu_set_last_error("bincount_weighted: unsupported weights dtype");
      return C_FAILED;
    }
  } else if (x->dtype == INSIGHT_DTYPE_I64) {
    switch (weights->dtype) {
    case INSIGHT_DTYPE_F32:
      DEFINE_BINCOUNT_WEIGHTED(int64_t, float);
      break;
    case INSIGHT_DTYPE_F64:
      DEFINE_BINCOUNT_WEIGHTED(int64_t, double);
      break;
    case INSIGHT_DTYPE_I32:
      DEFINE_BINCOUNT_WEIGHTED(int64_t, int32_t);
      break;
    case INSIGHT_DTYPE_I64:
      DEFINE_BINCOUNT_WEIGHTED(int64_t, int64_t);
      break;
    default:
      cpu_set_last_error("bincount_weighted: unsupported weights dtype");
      return C_FAILED;
    }
  } else {
    cpu_set_last_error("bincount_weighted: indices must be int32 or int64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(bincount_weighted, INSIGHT_DTYPE_I32,
                    bincount_weighted_kernel_cpu);
REGISTER_CPU_KERNEL(bincount_weighted, INSIGHT_DTYPE_I64,
                    bincount_weighted_kernel_cpu);
