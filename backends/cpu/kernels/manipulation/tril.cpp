// backends/cpu/kernels/manipulation/tril.cpp
/**
 * @file tril.cpp
 * @brief CPU kernel for tril operation (lower triangle).
 *
 * Returns the lower triangular part of a 2D array.
 *
 * @param inputs  [0] = InsightArray* output
 *                [1] = InsightArray* input
 *                [2] = int* k
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

// ============================================================================
// Tril helpers
// ============================================================================

/**
 * @brief Extract lower triangle of a 2D array.
 *
 * @tparam T Data type
 * @param src Source data (contiguous 2D)
 * @param dst Destination data
 * @param rows Number of rows
 * @param cols Number of columns
 * @param k Diagonal offset
 */
template <typename T>
static void cpu_tril_impl(const T *src, T *dst, int64_t rows, int64_t cols,
                          int64_t k) {

#pragma omp parallel for collapse(2)
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      int64_t idx = i * cols + j;
      if (j <= i + k) {
        dst[idx] = src[idx];
      } else {
        dst[idx] = 0;
      }
    }
  }
}

#ifdef __cplusplus
extern "C" {
#endif

C_Status tril_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("tril: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2]) {
    cpu_set_last_error("tril: k is null");
    return C_FAILED;
  }

  int k = *(int *)inputs[2];

  if (x->ndim != 2) {
    cpu_set_last_error("tril: input must be 2D");
    return C_FAILED;
  }

  int64_t rows = x->dims[0];
  int64_t cols = x->dims[1];

  // Ensure output is contiguous (should be by frontend)
  if (!insight_array_is_contiguous(out)) {
    cpu_set_last_error("tril: output must be contiguous");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_BOOL:
    cpu_tril_impl<bool>((const bool *)x->data, (bool *)out->data, rows, cols,
                        k);
    break;
  case INSIGHT_DTYPE_U8:
    cpu_tril_impl<uint8_t>((const uint8_t *)x->data, (uint8_t *)out->data, rows,
                           cols, k);
    break;
  case INSIGHT_DTYPE_I8:
    cpu_tril_impl<int8_t>((const int8_t *)x->data, (int8_t *)out->data, rows,
                          cols, k);
    break;
  case INSIGHT_DTYPE_I16:
    cpu_tril_impl<int16_t>((const int16_t *)x->data, (int16_t *)out->data, rows,
                           cols, k);
    break;
  case INSIGHT_DTYPE_I32:
    cpu_tril_impl<int32_t>((const int32_t *)x->data, (int32_t *)out->data, rows,
                           cols, k);
    break;
  case INSIGHT_DTYPE_I64:
    cpu_tril_impl<int64_t>((const int64_t *)x->data, (int64_t *)out->data, rows,
                           cols, k);
    break;
  case INSIGHT_DTYPE_U16:
    cpu_tril_impl<uint16_t>((const uint16_t *)x->data, (uint16_t *)out->data,
                            rows, cols, k);
    break;
  case INSIGHT_DTYPE_U32:
    cpu_tril_impl<uint32_t>((const uint32_t *)x->data, (uint32_t *)out->data,
                            rows, cols, k);
    break;
  case INSIGHT_DTYPE_U64:
    cpu_tril_impl<uint64_t>((const uint64_t *)x->data, (uint64_t *)out->data,
                            rows, cols, k);
    break;
  case INSIGHT_DTYPE_F32:
    cpu_tril_impl<float>((const float *)x->data, (float *)out->data, rows, cols,
                         k);
    break;
  case INSIGHT_DTYPE_F64:
    cpu_tril_impl<double>((const double *)x->data, (double *)out->data, rows,
                          cols, k);
    break;
  default:
    cpu_set_last_error("tril: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(tril, INSIGHT_DTYPE_BOOL, tril_kernel_cpu);
REGISTER_CPU_KERNEL(tril, INSIGHT_DTYPE_U8, tril_kernel_cpu);
REGISTER_CPU_KERNEL(tril, INSIGHT_DTYPE_I8, tril_kernel_cpu);
REGISTER_CPU_KERNEL(tril, INSIGHT_DTYPE_I16, tril_kernel_cpu);
REGISTER_CPU_KERNEL(tril, INSIGHT_DTYPE_I32, tril_kernel_cpu);
REGISTER_CPU_KERNEL(tril, INSIGHT_DTYPE_I64, tril_kernel_cpu);
REGISTER_CPU_KERNEL(tril, INSIGHT_DTYPE_U16, tril_kernel_cpu);
REGISTER_CPU_KERNEL(tril, INSIGHT_DTYPE_U32, tril_kernel_cpu);
REGISTER_CPU_KERNEL(tril, INSIGHT_DTYPE_U64, tril_kernel_cpu);
REGISTER_CPU_KERNEL(tril, INSIGHT_DTYPE_F32, tril_kernel_cpu);
REGISTER_CPU_KERNEL(tril, INSIGHT_DTYPE_F64, tril_kernel_cpu);
