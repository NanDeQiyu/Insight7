// backends/cpu/kernels/linalg/lu.cpp
/**
 * @file lu.cpp
 * @brief CPU kernel for LU decomposition with partial pivoting.
 *
 * Uses LAPACKE_{s,d}getrf when OpenBLAS is available, otherwise falls back to
 * a hand-optimized implementation.
 */

#include "../common/half_utils.h"
#include "common.h"
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================================
// Fallback implementation (used when OpenBLAS is not available)
// ============================================================================
#ifndef INSIGHT_USE_OPENBLAS

static void lu_f32_fallback(float *a, int *ipiv, int n) {
  for (int i = 0; i < n; ++i) {
    ipiv[i] = i;
  }

  for (int i = 0; i < n; ++i) {
    // Find pivot
    int pivot = i;
    float max_val = fabsf(a[i * n + i]);
    for (int k = i + 1; k < n; ++k) {
      float val = fabsf(a[k * n + i]);
      if (val > max_val) {
        max_val = val;
        pivot = k;
      }
    }

    if (max_val < 1e-12f) {
      continue;
    }

    if (pivot != i) {
      // Swap rows
      for (int j = 0; j < n; ++j) {
        float tmp = a[i * n + j];
        a[i * n + j] = a[pivot * n + j];
        a[pivot * n + j] = tmp;
      }
      int tmp = ipiv[i];
      ipiv[i] = ipiv[pivot];
      ipiv[pivot] = tmp;
    }

    float pivot_val = a[i * n + i];
    for (int k = i + 1; k < n; ++k) {
      a[k * n + i] /= pivot_val;
    }

    for (int k = i + 1; k < n; ++k) {
      float factor = a[k * n + i];
      for (int j = i + 1; j < n; ++j) {
        a[k * n + j] -= factor * a[i * n + j];
      }
    }
  }
}

static void lu_f64_fallback(double *a, int *ipiv, int n) {
  for (int i = 0; i < n; ++i) {
    ipiv[i] = i;
  }

  for (int i = 0; i < n; ++i) {
    int pivot = i;
    double max_val = fabs(a[i * n + i]);
    for (int k = i + 1; k < n; ++k) {
      double val = fabs(a[k * n + i]);
      if (val > max_val) {
        max_val = val;
        pivot = k;
      }
    }

    if (max_val < 1e-12) {
      continue;
    }

    if (pivot != i) {
      for (int j = 0; j < n; ++j) {
        double tmp = a[i * n + j];
        a[i * n + j] = a[pivot * n + j];
        a[pivot * n + j] = tmp;
      }
      int tmp = ipiv[i];
      ipiv[i] = ipiv[pivot];
      ipiv[pivot] = tmp;
    }

    double pivot_val = a[i * n + i];
    for (int k = i + 1; k < n; ++k) {
      a[k * n + i] /= pivot_val;
    }

    for (int k = i + 1; k < n; ++k) {
      double factor = a[k * n + i];
      for (int j = i + 1; j < n; ++j) {
        a[k * n + j] -= factor * a[i * n + j];
      }
    }
  }
}

#endif // !INSIGHT_USE_OPENBLAS

// ============================================================================
// LAPACKE-based implementation (used when OpenBLAS is available)
// ============================================================================
#ifdef INSIGHT_USE_OPENBLAS

/**
 * @brief Perform LU decomposition using LAPACK (single precision).
 * @param src Input matrix (row-major order)
 * @param a Output matrix (row-major order, contains L and U packed)
 * @param ipiv Pivot indices (1-based, will be filled)
 * @param n Matrix dimension
 */
static void lu_f32_lapack(const float *src, float *a, int *ipiv, int n) {
  // Allocate temporary column-major matrix for LAPACK
  float *colmajor = (float *)malloc(n * n * sizeof(float));
  if (!colmajor) {
    cpu_set_last_error("LU: memory allocation failed");
    return;
  }

  // Convert row-major to column-major (LAPACK expects column-major)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      colmajor[i + j * n] = src[i * n + j];
    }
  }

  // Perform LU decomposition (column-major)
  lapack_int info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, colmajor, n, ipiv);

  if (info != 0) {
    cpu_set_last_error("LU: LAPACKE_sgetrf failed");
    free(colmajor);
    return;
  }

  // Convert back from column-major to row-major for output
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i * n + j] = colmajor[i + j * n];
    }
  }

  free(colmajor);
}

/**
 * @brief Perform LU decomposition using LAPACK (double precision).
 * @param src Input matrix (row-major order)
 * @param a Output matrix (row-major order, contains L and U packed)
 * @param ipiv Pivot indices (1-based, will be filled)
 * @param n Matrix dimension
 */
static void lu_f64_lapack(const double *src, double *a, int *ipiv, int n) {
  // Allocate temporary column-major matrix for LAPACK
  double *colmajor = (double *)malloc(n * n * sizeof(double));
  if (!colmajor) {
    cpu_set_last_error("LU: memory allocation failed");
    return;
  }

  // Convert row-major to column-major (LAPACK expects column-major)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      colmajor[i + j * n] = src[i * n + j];
    }
  }

  // Perform LU decomposition (column-major)
  lapack_int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, colmajor, n, ipiv);

  if (info != 0) {
    cpu_set_last_error("LU: LAPACKE_dgetrf failed");
    free(colmajor);
    return;
  }

  // Convert back from column-major to row-major for output
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i * n + j] = colmajor[i + j * n];
    }
  }

  free(colmajor);
}

#endif // INSIGHT_USE_OPENBLAS

// ============================================================================
// Kernel entry point
// ============================================================================

C_Status lu_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *LU_arr = (InsightArray *)outputs[0];
  InsightArray *pivots_arr = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];
  int pivot_flag = *(int *)inputs[3]; // (unused, kept for compatibility)

  if (!LU_arr || !pivots_arr || !x) {
    cpu_set_last_error("lu: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(LU_arr) ||
      !cpu_ensure_contiguous(pivots_arr)) {
    return C_FAILED;
  }

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("lu: matrix must be square");
    return C_FAILED;
  }

  cpu_set_last_error("");

  int *ipiv = (int *)pivots_arr->data;

  if (x->dtype == INSIGHT_DTYPE_F16 || x->dtype == INSIGHT_DTYPE_BF16) {
    const uint16_t *x_src = (const uint16_t *)x->data;
    uint16_t *lu_dst = (uint16_t *)LU_arr->data;
    bool is_f16 = (x->dtype == INSIGHT_DTYPE_F16);
    int64_t total = (int64_t)n * n;
    std::vector<float> x_f32(total), lu_f32(total);
    if (is_f16) {
      for (int64_t i = 0; i < total; ++i)
        x_f32[i] = insight::f16_to_f32(x_src[i]);
    } else {
      for (int64_t i = 0; i < total; ++i)
        x_f32[i] = insight::bf16_to_f32(x_src[i]);
    }
#ifdef INSIGHT_USE_OPENBLAS
    lu_f32_lapack(x_f32.data(), lu_f32.data(), ipiv, n);
#else
    memcpy(lu_f32.data(), x_f32.data(), total * sizeof(float));
    lu_f32_fallback(lu_f32.data(), ipiv, n);
    for (int i = 0; i < n; ++i)
      ipiv[i] += 1;
#endif
    if (is_f16) {
      for (int64_t i = 0; i < total; ++i)
        lu_dst[i] = insight::f32_to_f16(lu_f32[i]);
    } else {
      for (int64_t i = 0; i < total; ++i)
        lu_dst[i] = insight::f32_to_bf16(lu_f32[i]);
    }
  } else

#ifdef INSIGHT_USE_OPENBLAS
      if (x->dtype == INSIGHT_DTYPE_F32) {
    lu_f32_lapack((const float *)x->data, (float *)LU_arr->data, ipiv, n);
  } else {
    lu_f64_lapack((const double *)x->data, (double *)LU_arr->data, ipiv, n);
  }
#else
    // Fallback: copy input to output first, then perform in-place decomposition
    memcpy(LU_arr->data, x->data, n * n * insight_dtype_size(x->dtype));
  if (x->dtype == INSIGHT_DTYPE_F32) {
    lu_f32_fallback((float *)LU_arr->data, ipiv, n);
  } else {
    lu_f64_fallback((double *)LU_arr->data, ipiv, n);
  }
#endif

  // Convert pivot indices to 1-based (LAPACK convention) for compatibility
  // Note: LAPACKE already returns 1-based indices, fallback returns 0-based
#ifdef INSIGHT_USE_OPENBLAS
  // LAPACKE already returns 1-based, no conversion needed
#else
  for (int i = 0; i < n; ++i) {
    ipiv[i] += 1;
  }
#endif

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(lu, INSIGHT_DTYPE_F32, lu_kernel_cpu);
REGISTER_CPU_KERNEL(lu, INSIGHT_DTYPE_F64, lu_kernel_cpu);
REGISTER_CPU_KERNEL(lu, INSIGHT_DTYPE_F16, lu_kernel_cpu);
REGISTER_CPU_KERNEL(lu, INSIGHT_DTYPE_BF16, lu_kernel_cpu);
