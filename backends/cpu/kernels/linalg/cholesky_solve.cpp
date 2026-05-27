// backends/cpu/kernels/linalg/cholesky_solve.cpp
/**
 * @file cholesky_solve.cpp
 * @brief CPU kernel for solving linear systems using Cholesky factor.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <cstdlib>
#include <cstring>

static void cholesky_solve_f32(const float *A, const float *B, float *X, int n,
                               int nrhs, int lower) {
  float *a = (float *)malloc(n * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("cholesky_solve: memory allocation failed");
    return;
  }
  memcpy(a, A, n * n * sizeof(float));

  float *b = (float *)malloc(n * nrhs * sizeof(float));
  if (!b) {
    cpu_set_last_error("cholesky_solve: memory allocation failed");
    free(a);
    return;
  }
  memcpy(b, B, n * nrhs * sizeof(float));

  int matrix_layout = LAPACK_COL_MAJOR;
  char uplo = lower ? 'L' : 'U';
  lapack_int info = LAPACKE_spotrs(matrix_layout, uplo, n, nrhs, a, n, b, n);

  if (info != 0) {
    cpu_set_last_error("cholesky_solve: LAPACKE_spotrs failed");
  } else {
    memcpy(X, b, n * nrhs * sizeof(float));
  }

  free(a);
  free(b);
}

static void cholesky_solve_f64(const double *A, const double *B, double *X,
                               int n, int nrhs, int lower) {
  double *a = (double *)malloc(n * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("cholesky_solve: memory allocation failed");
    return;
  }
  memcpy(a, A, n * n * sizeof(double));

  double *b = (double *)malloc(n * nrhs * sizeof(double));
  if (!b) {
    cpu_set_last_error("cholesky_solve: memory allocation failed");
    free(a);
    return;
  }
  memcpy(b, B, n * nrhs * sizeof(double));

  int matrix_layout = LAPACK_COL_MAJOR;
  char uplo = lower ? 'L' : 'U';
  lapack_int info = LAPACKE_dpotrs(matrix_layout, uplo, n, nrhs, a, n, b, n);

  if (info != 0) {
    cpu_set_last_error("cholesky_solve: LAPACKE_dpotrs failed");
  } else {
    memcpy(X, b, n * nrhs * sizeof(double));
  }

  free(a);
  free(b);
}

C_Status cholesky_solve_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *A = (InsightArray *)inputs[1];
  InsightArray *B = (InsightArray *)inputs[2];
  int lower = *(int *)inputs[3];

  if (!out || !A || !B) {
    cpu_set_last_error("cholesky_solve: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(A) || !cpu_ensure_contiguous(B) ||
      !cpu_ensure_contiguous(out)) {
    return C_FAILED;
  }

  int n = (int)A->dims[0];
  int nrhs = (B->ndim == 1) ? 1 : (int)B->dims[1];

  if (A->dims[1] != n) {
    cpu_set_last_error("cholesky_solve: A must be square");
    return C_FAILED;
  }

  cpu_set_last_error("");

  if (A->dtype == INSIGHT_DTYPE_F32) {
    cholesky_solve_f32((float *)A->data, (float *)B->data, (float *)out->data,
                       n, nrhs, lower);
  } else {
    cholesky_solve_f64((double *)A->data, (double *)B->data,
                       (double *)out->data, n, nrhs, lower);
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(cholesky_solve, INSIGHT_DTYPE_F32,
                    cholesky_solve_kernel_cpu);
REGISTER_CPU_KERNEL(cholesky_solve, INSIGHT_DTYPE_F64,
                    cholesky_solve_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status cholesky_solve_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("cholesky_solve: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(cholesky_solve, INSIGHT_DTYPE_F32,
                    cholesky_solve_kernel_cpu);
REGISTER_CPU_KERNEL(cholesky_solve, INSIGHT_DTYPE_F64,
                    cholesky_solve_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
