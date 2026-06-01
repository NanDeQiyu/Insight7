// backends/cpu/kernels/linalg/qr.cpp
/**
 * @file qr.cpp
 * @brief CPU kernel for QR decomposition.
 *
 * Uses LAPACKE_{s,d}geqrf and LAPACKE_{s,d}orgqr when OpenBLAS is available,
 * otherwise falls back to a hand-optimized Householder implementation.
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

#include <algorithm>
#include <cmath>
#include <vector>

template <typename T>
static void qr_impl_fallback(const T *src, T *q, T *r, int m, int n, int mode) {
  int k = std::min(m, n);

  // Copy matrix to column-major (we will modify it)
  std::vector<T> A(m * n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i + j * m] = src[i * n + j];
    }
  }

  std::vector<T> tau(k);

  // QR decomposition: Householder transformations column by column
  for (int j = 0; j < k; ++j) {
    int len = m - j;
    T *col_ptr = A.data() + j * m + j;

    // Compute norm of current column
    T norm = 0;
    for (int i = 0; i < len; ++i) {
      norm += col_ptr[i] * col_ptr[i];
    }
    norm = std::sqrt(norm);

    if (norm > std::numeric_limits<T>::epsilon()) {
      T &first = col_ptr[0];
      T sign = (first > 0) ? T(1) : T(-1);
      T u1 = first + sign * norm;
      tau[j] = sign * u1 / norm;

      first = -sign * norm;

      T inv_u1 = T(1) / u1;
      for (int i = 1; i < len; ++i) {
        col_ptr[i] *= inv_u1;
      }

      // Apply to remaining columns
      for (int col = j + 1; col < n; ++col) {
        T *target_col = A.data() + col * m + j;

        T dot = target_col[0];
        for (int row = 1; row < len; ++row) {
          dot += col_ptr[row] * target_col[row];
        }
        dot *= tau[j];

        target_col[0] -= dot;
        for (int row = 1; row < len; ++row) {
          target_col[row] -= dot * col_ptr[row];
        }
      }
    } else {
      tau[j] = T(0);
    }
  }

  // Extract R matrix
  std::vector<T> R_colmajor(k * n, T(0));
  for (int i = 0; i < k; ++i) {
    for (int j = i; j < n; ++j) {
      R_colmajor[i + j * k] = A[i + j * m];
    }
  }

  // Build Q matrix (start with identity)
  std::vector<T> Q_colmajor(m * m, T(0));
  for (int i = 0; i < m; ++i) {
    Q_colmajor[i + i * m] = T(1);
  }

  // Apply Householder transformations in reverse order
  for (int j = k - 1; j >= 0; --j) {
    int len = m - j;
    T *v_ptr = A.data() + j * m + j;
    v_ptr[0] = T(1);

    for (int col = 0; col < m; ++col) {
      T *Q_col = Q_colmajor.data() + col * m + j;

      T dot = Q_col[0];
      for (int row = 1; row < len; ++row) {
        dot += v_ptr[row] * Q_col[row];
      }
      dot *= tau[j];

      Q_col[0] -= dot;
      for (int row = 1; row < len; ++row) {
        Q_col[row] -= dot * v_ptr[row];
      }
    }
  }

  // Convert results to row-major
  if (mode == 0) { // complete Q
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        q[i * m + j] = Q_colmajor[i + j * m];
      }
    }
  } else { // reduced Q
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < k; ++j) {
        q[i * k + j] = Q_colmajor[i + j * m];
      }
    }
  }

  // Convert R to row-major
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      r[i * n + j] = R_colmajor[i + j * k];
    }
  }
}

static void qr_f64_fallback(const double *src, double *q, double *r, int m,
                            int n, int mode) {
  qr_impl_fallback(src, q, r, m, n, mode);
}

static void qr_f32_fallback(const float *src, float *q, float *r, int m, int n,
                            int mode) {
  qr_impl_fallback(src, q, r, m, n, mode);
}

#endif // !INSIGHT_USE_OPENBLAS

// ============================================================================
// LAPACKE-based implementation (used when OpenBLAS is available)
// ============================================================================
#ifdef INSIGHT_USE_OPENBLAS

#include <cstdlib>
#include <cstring>

static void qr_f64_lapack(const double *src, double *q, double *r, int m, int n,
                          int mode) {
  int k = m < n ? m : n;
  int ld = m;

  // Copy to column major order
  double *a = (double *)malloc(m * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("qr: memory allocation failed");
    return;
  }
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * ld] = src[i * n + j];
    }
  }

  double *tau = (double *)malloc(k * sizeof(double));
  if (!tau) {
    cpu_set_last_error("qr: memory allocation failed");
    free(a);
    return;
  }

  // QR decomposition (LAPACKE will automatically manage the workspace)
  int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a, ld, tau);
  if (info != 0) {
    cpu_set_last_error("qr: LAPACK dgeqrf failed");
    free(a);
    free(tau);
    return;
  }

  // Extract R (upper triangle)
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      r[i * n + j] = (j >= i) ? a[i + j * ld] : 0.0;
    }
  }

  // Generate Q
  int q_cols = (mode == 0) ? m : k;
  info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, q_cols, k, a, ld, tau);
  if (info != 0) {
    cpu_set_last_error("qr: LAPACK dorgqr failed");
    free(a);
    free(tau);
    return;
  }

  // Output Q (convert from column-major order to row-major order)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < q_cols; ++j) {
      q[i * q_cols + j] = a[i + j * ld];
    }
  }

  free(a);
  free(tau);
}

static void qr_f32_lapack(const float *src, float *q, float *r, int m, int n,
                          int mode) {
  int k = m < n ? m : n;
  int ld = m;

  float *a = (float *)malloc(m * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("qr: memory allocation failed");
    return;
  }
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * ld] = src[i * n + j];
    }
  }

  float *tau = (float *)malloc(k * sizeof(float));
  if (!tau) {
    cpu_set_last_error("qr: memory allocation failed");
    free(a);
    return;
  }

  int info = LAPACKE_sgeqrf(LAPACK_COL_MAJOR, m, n, a, ld, tau);
  if (info != 0) {
    cpu_set_last_error("qr: LAPACK sgeqrf failed");
    free(a);
    free(tau);
    return;
  }

  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      r[i * n + j] = (j >= i) ? a[i + j * ld] : 0.0f;
    }
  }

  int q_cols = (mode == 0) ? m : k;
  info = LAPACKE_sorgqr(LAPACK_COL_MAJOR, m, q_cols, k, a, ld, tau);
  if (info != 0) {
    cpu_set_last_error("qr: LAPACK sorgqr failed");
    free(a);
    free(tau);
    return;
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < q_cols; ++j) {
      q[i * q_cols + j] = a[i + j * ld];
    }
  }

  free(a);
  free(tau);
}

#endif // INSIGHT_USE_OPENBLAS

// ============================================================================
// Kernel entry point
// ============================================================================
extern "C" {

C_Status qr_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *Q = (InsightArray *)outputs[0];
  InsightArray *R = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];
  int mode = *(int *)inputs[3];

  if (!Q || !R || !x) {
    cpu_set_last_error("qr: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(x)) {
    return C_FAILED;
  }

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  cpu_set_last_error("");

  if (x->dtype == INSIGHT_DTYPE_F16 || x->dtype == INSIGHT_DTYPE_BF16) {
    const uint16_t *x_src = (const uint16_t *)x->data;
    uint16_t *q_dst = (uint16_t *)Q->data;
    uint16_t *r_dst = (uint16_t *)R->data;
    bool is_f16 = (x->dtype == INSIGHT_DTYPE_F16);
    int k = m < n ? m : n;
    int q_cols = (mode == 0) ? m : k;
    std::vector<float> x_f32((int64_t)m * n);
    std::vector<float> q_f32((int64_t)m * q_cols);
    std::vector<float> r_f32((int64_t)k * n);
    if (is_f16) {
      for (int64_t i = 0; i < (int64_t)m * n; ++i)
        x_f32[i] = insight::f16_to_f32(x_src[i]);
    } else {
      for (int64_t i = 0; i < (int64_t)m * n; ++i)
        x_f32[i] = insight::bf16_to_f32(x_src[i]);
    }
#ifdef INSIGHT_USE_OPENBLAS
    qr_f32_lapack(x_f32.data(), q_f32.data(), r_f32.data(), m, n, mode);
#else
    qr_f32_fallback(x_f32.data(), q_f32.data(), r_f32.data(), m, n, mode);
#endif
    if (is_f16) {
      for (int64_t i = 0; i < (int64_t)m * q_cols; ++i)
        q_dst[i] = insight::f32_to_f16(q_f32[i]);
      for (int64_t i = 0; i < (int64_t)k * n; ++i)
        r_dst[i] = insight::f32_to_f16(r_f32[i]);
    } else {
      for (int64_t i = 0; i < (int64_t)m * q_cols; ++i)
        q_dst[i] = insight::f32_to_bf16(q_f32[i]);
      for (int64_t i = 0; i < (int64_t)k * n; ++i)
        r_dst[i] = insight::f32_to_bf16(r_f32[i]);
    }
  } else
#ifdef INSIGHT_USE_OPENBLAS
      if (x->dtype == INSIGHT_DTYPE_F64) {
    qr_f64_lapack((double *)x->data, (double *)Q->data, (double *)R->data, m, n,
                  mode);
  } else if (x->dtype == INSIGHT_DTYPE_F32) {
    qr_f32_lapack((float *)x->data, (float *)Q->data, (float *)R->data, m, n,
                  mode);
  } else {
    cpu_set_last_error("qr: unsupported dtype (only F32/F64)");
    return C_FAILED;
  }
#else
      if (x->dtype == INSIGHT_DTYPE_F64) {
    qr_f64_fallback((double *)x->data, (double *)Q->data, (double *)R->data, m,
                    n, mode);
  } else if (x->dtype == INSIGHT_DTYPE_F32) {
    qr_f32_fallback((float *)x->data, (float *)Q->data, (float *)R->data, m, n,
                    mode);
  } else {
    cpu_set_last_error("qr: unsupported dtype (only F32/F64)");
    return C_FAILED;
  }
#endif

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(qr, INSIGHT_DTYPE_F64, qr_kernel_cpu);
REGISTER_CPU_KERNEL(qr, INSIGHT_DTYPE_F32, qr_kernel_cpu);
REGISTER_CPU_KERNEL(qr, INSIGHT_DTYPE_F16, qr_kernel_cpu);
REGISTER_CPU_KERNEL(qr, INSIGHT_DTYPE_BF16, qr_kernel_cpu);
