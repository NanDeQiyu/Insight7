// backends/cpu/kernels/linalg/lstsq.cpp
/**
 * @file lstsq.cpp
 * @brief CPU kernel for least squares solution using LAPACK.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <cstdlib>
#include <cstring>

static void lstsq_f64(const double *A, const double *B, double *X, int m, int n,
                      int nrhs) {
  // 复制 A 到列主序
  double *a = (double *)malloc(m * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("lstsq: memory allocation failed");
    return;
  }
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * m] = A[i * n + j];
    }
  }

  // 复制 B 到列主序，并扩展行数到 max(m, n)
  int b_rows = (m > n) ? m : n;
  double *b = (double *)malloc(b_rows * nrhs * sizeof(double));
  if (!b) {
    cpu_set_last_error("lstsq: memory allocation failed");
    free(a);
    return;
  }
  memset(b, 0, b_rows * nrhs * sizeof(double));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < nrhs; ++j) {
      b[i + j * b_rows] = B[i * nrhs + j];
    }
  }

  // LAPACKE_dgels 内部自动管理 workspace
  int info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', m, n, nrhs, a, m, b, b_rows);

  if (info != 0) {
    cpu_set_last_error("lstsq: LAPACK dgels failed");
    free(a);
    free(b);
    return;
  }

  // 提取解 (前 n 行)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < nrhs; ++j) {
      X[i * nrhs + j] = b[i + j * b_rows];
    }
  }

  free(a);
  free(b);
}

static void lstsq_f32(const float *A, const float *B, float *X, int m, int n,
                      int nrhs) {
  float *a = (float *)malloc(m * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("lstsq: memory allocation failed");
    return;
  }
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * m] = A[i * n + j];
    }
  }

  int b_rows = (m > n) ? m : n;
  float *b = (float *)malloc(b_rows * nrhs * sizeof(float));
  if (!b) {
    cpu_set_last_error("lstsq: memory allocation failed");
    free(a);
    return;
  }
  memset(b, 0, b_rows * nrhs * sizeof(float));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < nrhs; ++j) {
      b[i + j * b_rows] = B[i * nrhs + j];
    }
  }

  int info = LAPACKE_sgels(LAPACK_COL_MAJOR, 'N', m, n, nrhs, a, m, b, b_rows);

  if (info != 0) {
    cpu_set_last_error("lstsq: LAPACK sgels failed");
    free(a);
    free(b);
    return;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < nrhs; ++j) {
      X[i * nrhs + j] = b[i + j * b_rows];
    }
  }

  free(a);
  free(b);
}

C_Status lstsq_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *A = (InsightArray *)inputs[1];
  InsightArray *B = (InsightArray *)inputs[2];
  double rcond = *(double *)inputs[3]; // not used, kept for compatibility

  if (!out || !A || !B) {
    cpu_set_last_error("lstsq: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(A) || !cpu_ensure_contiguous(B) ||
      !cpu_ensure_contiguous(out)) {
    return C_FAILED;
  }

  int m = (int)A->dims[0];
  int n = (int)A->dims[1];
  int nrhs = (B->ndim == 1) ? 1 : (int)B->dims[1];

  cpu_set_last_error("");

  if (A->dtype == INSIGHT_DTYPE_F32) {
    lstsq_f32((float *)A->data, (float *)B->data, (float *)out->data, m, n,
              nrhs);
  } else {
    lstsq_f64((double *)A->data, (double *)B->data, (double *)out->data, m, n,
              nrhs);
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(lstsq, INSIGHT_DTYPE_F32, lstsq_kernel_cpu);
REGISTER_CPU_KERNEL(lstsq, INSIGHT_DTYPE_F64, lstsq_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status lstsq_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("lstsq: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(lstsq, INSIGHT_DTYPE_F32, lstsq_kernel_cpu);
REGISTER_CPU_KERNEL(lstsq, INSIGHT_DTYPE_F64, lstsq_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
