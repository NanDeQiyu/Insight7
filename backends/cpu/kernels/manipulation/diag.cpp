// backends/cpu/kernels/manipulation/diag.cpp
/**
 * @file diag.cpp
 * @brief CPU kernel for diag operation.
 *
 * Extracts diagonal or constructs diagonal matrix.
 *
 * @param inputs  [0] = InsightArray* output
 *                [1] = InsightArray* input
 *                [2] = int* k
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

// ============================================================================
// Diag helpers
// ============================================================================

/**
 * @brief Extract diagonal from 2D array.
 *
 * @tparam T Data type
 * @param src Source data (contiguous 2D)
 * @param dst Destination diagonal
 * @param rows Number of rows
 * @param cols Number of columns
 * @param k Diagonal offset
 */
template <typename T>
static void cpu_diag_extract_impl(const T *src, T *dst, int64_t rows,
                                  int64_t cols, int64_t k) {

  int64_t diag_len;
  if (k >= 0) {
    diag_len = (rows < cols - k) ? rows : cols - k;
  } else {
    diag_len = (rows + k < cols) ? rows + k : cols;
  }

#pragma omp parallel for
  for (int64_t i = 0; i < diag_len; ++i) {
    int64_t src_idx;
    if (k >= 0) {
      src_idx = i * cols + (i + k);
    } else {
      src_idx = (i - k) * cols + i;
    }
    dst[i] = src[src_idx];
  }
}

/**
 * @brief Construct diagonal matrix from 1D array.
 *
 * @tparam T Data type
 * @param src Source diagonal
 * @param dst Destination matrix
 * @param size Matrix size
 * @param k Diagonal offset
 */
template <typename T>
static void cpu_diag_construct_impl(const T *src, T *dst, int64_t size,
                                    int64_t k) {

  int64_t n = size;
  int64_t cols = size + (k > 0 ? k : -k);

  // Fill with zeros
#pragma omp parallel for
  for (int64_t i = 0; i < n * cols; ++i) {
    dst[i] = 0;
  }

  // Set diagonal
#pragma omp parallel for
  for (int64_t i = 0; i < n; ++i) {
    int64_t j = i + k;
    if (j >= 0 && j < cols) {
      dst[i * cols + j] = src[i];
    }
  }
}

#ifdef __cplusplus
extern "C" {
#endif

C_Status diag_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("diag: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2]) {
    cpu_set_last_error("diag: k is null");
    return C_FAILED;
  }

  int k = *(int *)inputs[2];

  // Ensure output is contiguous (should be by frontend)
  if (!insight_array_is_contiguous(out)) {
    cpu_set_last_error("diag: output must be contiguous");
    return C_FAILED;
  }

  if (x->ndim == 1) {
    // Construct diagonal matrix from 1D array
    int64_t n = x->numel;
    int64_t size = n + (k > 0 ? k : -k);

    switch (x->dtype) {
    case INSIGHT_DTYPE_BOOL:
      cpu_diag_construct_impl<bool>((const bool *)x->data, (bool *)out->data,
                                    size, k);
      break;
    case INSIGHT_DTYPE_U8:
      cpu_diag_construct_impl<uint8_t>((const uint8_t *)x->data,
                                       (uint8_t *)out->data, size, k);
      break;
    case INSIGHT_DTYPE_I8:
      cpu_diag_construct_impl<int8_t>((const int8_t *)x->data,
                                      (int8_t *)out->data, size, k);
      break;
    case INSIGHT_DTYPE_I16:
      cpu_diag_construct_impl<int16_t>((const int16_t *)x->data,
                                       (int16_t *)out->data, size, k);
      break;
    case INSIGHT_DTYPE_I32:
      cpu_diag_construct_impl<int32_t>((const int32_t *)x->data,
                                       (int32_t *)out->data, size, k);
      break;
    case INSIGHT_DTYPE_I64:
      cpu_diag_construct_impl<int64_t>((const int64_t *)x->data,
                                       (int64_t *)out->data, size, k);
      break;
    case INSIGHT_DTYPE_U16:
      cpu_diag_construct_impl<uint16_t>((const uint16_t *)x->data,
                                        (uint16_t *)out->data, size, k);
      break;
    case INSIGHT_DTYPE_U32:
      cpu_diag_construct_impl<uint32_t>((const uint32_t *)x->data,
                                        (uint32_t *)out->data, size, k);
      break;
    case INSIGHT_DTYPE_U64:
      cpu_diag_construct_impl<uint64_t>((const uint64_t *)x->data,
                                        (uint64_t *)out->data, size, k);
      break;
    case INSIGHT_DTYPE_F32:
      cpu_diag_construct_impl<float>((const float *)x->data, (float *)out->data,
                                     size, k);
      break;
    case INSIGHT_DTYPE_F64:
      cpu_diag_construct_impl<double>((const double *)x->data,
                                      (double *)out->data, size, k);
      break;
    default:
      cpu_set_last_error("diag: unsupported dtype");
      return C_FAILED;
    }
  } else if (x->ndim == 2) {
    // Extract diagonal from 2D array
    int64_t rows = x->dims[0];
    int64_t cols = x->dims[1];

    switch (x->dtype) {
    case INSIGHT_DTYPE_BOOL:
      cpu_diag_extract_impl<bool>((const bool *)x->data, (bool *)out->data,
                                  rows, cols, k);
      break;
    case INSIGHT_DTYPE_U8:
      cpu_diag_extract_impl<uint8_t>((const uint8_t *)x->data,
                                     (uint8_t *)out->data, rows, cols, k);
      break;
    case INSIGHT_DTYPE_I8:
      cpu_diag_extract_impl<int8_t>((const int8_t *)x->data,
                                    (int8_t *)out->data, rows, cols, k);
      break;
    case INSIGHT_DTYPE_I16:
      cpu_diag_extract_impl<int16_t>((const int16_t *)x->data,
                                     (int16_t *)out->data, rows, cols, k);
      break;
    case INSIGHT_DTYPE_I32:
      cpu_diag_extract_impl<int32_t>((const int32_t *)x->data,
                                     (int32_t *)out->data, rows, cols, k);
      break;
    case INSIGHT_DTYPE_I64:
      cpu_diag_extract_impl<int64_t>((const int64_t *)x->data,
                                     (int64_t *)out->data, rows, cols, k);
      break;
    case INSIGHT_DTYPE_U16:
      cpu_diag_extract_impl<uint16_t>((const uint16_t *)x->data,
                                      (uint16_t *)out->data, rows, cols, k);
      break;
    case INSIGHT_DTYPE_U32:
      cpu_diag_extract_impl<uint32_t>((const uint32_t *)x->data,
                                      (uint32_t *)out->data, rows, cols, k);
      break;
    case INSIGHT_DTYPE_U64:
      cpu_diag_extract_impl<uint64_t>((const uint64_t *)x->data,
                                      (uint64_t *)out->data, rows, cols, k);
      break;
    case INSIGHT_DTYPE_F32:
      cpu_diag_extract_impl<float>((const float *)x->data, (float *)out->data,
                                   rows, cols, k);
      break;
    case INSIGHT_DTYPE_F64:
      cpu_diag_extract_impl<double>((const double *)x->data,
                                    (double *)out->data, rows, cols, k);
      break;
    default:
      cpu_set_last_error("diag: unsupported dtype");
      return C_FAILED;
    }
  } else {
    cpu_set_last_error("diag: input must be 1D or 2D");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(diag, INSIGHT_DTYPE_BOOL, diag_kernel_cpu);
REGISTER_CPU_KERNEL(diag, INSIGHT_DTYPE_U8, diag_kernel_cpu);
REGISTER_CPU_KERNEL(diag, INSIGHT_DTYPE_I8, diag_kernel_cpu);
REGISTER_CPU_KERNEL(diag, INSIGHT_DTYPE_I16, diag_kernel_cpu);
REGISTER_CPU_KERNEL(diag, INSIGHT_DTYPE_I32, diag_kernel_cpu);
REGISTER_CPU_KERNEL(diag, INSIGHT_DTYPE_I64, diag_kernel_cpu);
REGISTER_CPU_KERNEL(diag, INSIGHT_DTYPE_U16, diag_kernel_cpu);
REGISTER_CPU_KERNEL(diag, INSIGHT_DTYPE_U32, diag_kernel_cpu);
REGISTER_CPU_KERNEL(diag, INSIGHT_DTYPE_U64, diag_kernel_cpu);
REGISTER_CPU_KERNEL(diag, INSIGHT_DTYPE_F32, diag_kernel_cpu);
REGISTER_CPU_KERNEL(diag, INSIGHT_DTYPE_F64, diag_kernel_cpu);
