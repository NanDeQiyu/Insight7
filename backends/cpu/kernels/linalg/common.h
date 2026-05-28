// backends/cpu/kernels/linalg/common.h
/**
 * @file common.h
 * @brief Common utilities for CPU linear algebra kernels.
 */

#ifndef CPU_LINALG_COMMON_H
#define CPU_LINALG_COMMON_H

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cstdio>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Platform/library configuration
// ============================================================================

#ifdef INSIGHT_USE_OPENBLAS
#include <cblas.h>
#include <lapacke.h> // LAPACK C Interface
#endif

// ============================================================================
// Memory layout conversion (row major ? column major)
// ============================================================================

/**
 * @brief Convert row-major to column-major (float) with multi-threading
 */
static inline void cpu_rowmajor_to_colmajor_f32(const float *src, float *dst,
                                                int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      dst[i + j * rows] = src[i * cols + j];
    }
  }
}

/**
 * @brief Convert row-major to column-major (double) with multi-threading
 */
static inline void cpu_rowmajor_to_colmajor_f64(const double *src, double *dst,
                                                int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      dst[i + j * rows] = src[i * cols + j];
    }
  }
}

/**
 * @brief Convert column-major to row-major (float) with multi-threading
 */
static inline void cpu_colmajor_to_rowmajor_f32(const float *src, float *dst,
                                                int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      dst[i * cols + j] = src[i + j * rows];
    }
  }
}

/**
 * @brief Convert column-major to row-major (double) with multi-threading
 */
static inline void cpu_colmajor_to_rowmajor_f64(const double *src, double *dst,
                                                int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      dst[i * cols + j] = src[i + j * rows];
    }
  }
}

// ============================================================================
// LAPACK error handling
// ============================================================================

/**
 * @brief Check LAPACK info code and set error message if needed
 */
static inline void cpu_check_lapack_info(int info, const char *func_name) {
  if (info < 0) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: illegal value at argument %d", func_name,
             -info);
    cpu_set_last_error(msg);
  } else if (info > 0) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "%s: singular/not positive definite (U(%d,%d) is zero)", func_name,
             info, info);
    cpu_set_last_error(msg);
  }
}

// ============================================================================
// Make sure array is contiguous (simple implementation, just check)
// ============================================================================

/**
 * @brief Check if array is contiguous; if not, set error and return 0.
 */
static inline int cpu_ensure_contiguous(InsightArray *arr) {
  if (insight_array_is_contiguous(arr))
    return 1;
  cpu_set_last_error("non-contiguous array not supported for this operation");
  return 0;
}

#endif // CPU_LINALG_COMMON_H
