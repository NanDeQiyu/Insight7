// backends/cpu/kernels/linalg/eigvals.cpp
/**
 * @file eigvals.cpp
 * @brief CPU kernel for eigenvalues only (complex).
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <cstdlib>
#include <cstring>

static void eigvals_f32(const float *src, float *vals_real, float *vals_imag,
                        int n) {
  float *a = (float *)malloc(n * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("eigvals: memory allocation failed");
    return;
  }
  memcpy(a, src, n * n * sizeof(float));

  float *wr = (float *)malloc(n * sizeof(float));
  float *wi = (float *)malloc(n * sizeof(float));
  float *work = (float *)malloc(1 * sizeof(float));
  int lwork = -1;

  if (!wr || !wi || !work) {
    cpu_set_last_error("eigvals: memory allocation failed");
    free(a);
    free(wr);
    free(wi);
    free(work);
    return;
  }

  // workspace query
  LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'N', n, a, n, wr, wi, NULL, 1, NULL, 1);

  lwork = (int)work[0];
  float *work_opt = (float *)realloc(work, lwork * sizeof(float));
  if (!work_opt) {
    cpu_set_last_error("eigvals: memory allocation failed");
    free(a);
    free(wr);
    free(wi);
    free(work);
    return;
  }
  work = work_opt;

  LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'N', n, a, n, wr, wi, NULL, 1, NULL, 1);

  for (int i = 0; i < n; ++i) {
    vals_real[i] = wr[i];
    vals_imag[i] = wi[i];
  }

  free(a);
  free(wr);
  free(wi);
  free(work);
}

static void eigvals_f64(const double *src, double *vals_real, double *vals_imag,
                        int n) {
  double *a = (double *)malloc(n * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("eigvals: memory allocation failed");
    return;
  }
  memcpy(a, src, n * n * sizeof(double));

  double *wr = (double *)malloc(n * sizeof(double));
  double *wi = (double *)malloc(n * sizeof(double));
  double *work = (double *)malloc(1 * sizeof(double));
  int lwork = -1;

  if (!wr || !wi || !work) {
    cpu_set_last_error("eigvals: memory allocation failed");
    free(a);
    free(wr);
    free(wi);
    free(work);
    return;
  }

  // workspace query
  LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', n, a, n, wr, wi, NULL, 1, NULL, 1);

  lwork = (int)work[0];
  double *work_opt = (double *)realloc(work, lwork * sizeof(double));
  if (!work_opt) {
    cpu_set_last_error("eigvals: memory allocation failed");
    free(a);
    free(wr);
    free(wi);
    free(work);
    return;
  }
  work = work_opt;

  LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', n, a, n, wr, wi, NULL, 1, NULL, 1);

  for (int i = 0; i < n; ++i) {
    vals_real[i] = wr[i];
    vals_imag[i] = wi[i];
  }

  free(a);
  free(wr);
  free(wi);
  free(work);
}

C_Status eigvals_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out_real = (InsightArray *)outputs[0];
  InsightArray *out_imag = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];

  if (!out_real || !out_imag || !x) {
    cpu_set_last_error("eigvals: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(out_real) ||
      !cpu_ensure_contiguous(out_imag)) {
    return C_FAILED;
  }

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("eigvals: matrix must be square");
    return C_FAILED;
  }

  cpu_set_last_error("");

  if (x->dtype == INSIGHT_DTYPE_F32) {
    eigvals_f32((float *)x->data, (float *)out_real->data,
                (float *)out_imag->data, n);
  } else {
    eigvals_f64((double *)x->data, (double *)out_real->data,
                (double *)out_imag->data, n);
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(eigvals, INSIGHT_DTYPE_F32, eigvals_kernel_cpu);
REGISTER_CPU_KERNEL(eigvals, INSIGHT_DTYPE_F64, eigvals_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status eigvals_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("eigvals: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(eigvals, INSIGHT_DTYPE_F32, eigvals_kernel_cpu);
REGISTER_CPU_KERNEL(eigvals, INSIGHT_DTYPE_F64, eigvals_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
