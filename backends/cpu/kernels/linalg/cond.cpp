// backends/cpu/kernels/linalg/cond.cpp
/**
 * @file cond.cpp
 * @brief CPU kernel for condition number of a matrix.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <cstdlib>
#include <cstring>

static double cond_f64(const double *src, int m, int n) {
  double *a = (double *)malloc(m * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("cond: memory allocation failed");
    return 0.0;
  }
  memcpy(a, src, m * n * sizeof(double));

  int *ipiv = (int *)malloc(n * sizeof(int));
  if (!ipiv) {
    cpu_set_last_error("cond: memory allocation failed");
    free(a);
    return 0.0;
  }

  int info;
  info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, a, m, ipiv);

  if (info != 0) {
    cpu_set_last_error("cond: LAPACKE_dgetrf failed");
    free(a);
    free(ipiv);
    return 0.0;
  }

  char norm = '1';
  double anorm = LAPACKE_dlange(LAPACK_COL_MAJOR, norm, m, n, src, m);

  double rcond;
  double *work = (double *)malloc(4 * n * sizeof(double));
  int *iwork = (int *)malloc(n * sizeof(int));

  if (!work || !iwork) {
    cpu_set_last_error("cond: memory allocation failed");
    free(a);
    free(ipiv);
    free(work);
    free(iwork);
    return 0.0;
  }

  LAPACKE_dgecon(LAPACK_COL_MAJOR, norm, n, a, m, anorm, &rcond);

  free(a);
  free(ipiv);
  free(work);
  free(iwork);

  if (info != 0) {
    return 0.0;
  }
  return 1.0 / rcond;
}

static float cond_f32(const float *src, int m, int n) {
  double *src_f64 = (double *)malloc(m * n * sizeof(double));
  if (!src_f64) {
    cpu_set_last_error("cond: memory allocation failed");
    return 0.0f;
  }
  for (int i = 0; i < m * n; ++i) {
    src_f64[i] = src[i];
  }

  double result_f64 = cond_f64(src_f64, m, n);

  free(src_f64);
  return (float)result_f64;
}

C_Status cond_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  double p = *(double *)inputs[2]; // unused for 1-norm

  if (!out || !x) {
    cpu_set_last_error("cond: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(x)) {
    return C_FAILED;
  }

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  cpu_set_last_error("");

  if (x->dtype == INSIGHT_DTYPE_F32) {
    float *dst = (float *)out->data;
    *dst = cond_f32((float *)x->data, m, n);
  } else {
    double *dst = (double *)out->data;
    *dst = cond_f64((double *)x->data, m, n);
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(cond, INSIGHT_DTYPE_F32, cond_kernel_cpu);
REGISTER_CPU_KERNEL(cond, INSIGHT_DTYPE_F64, cond_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status cond_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("cond: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(cond, INSIGHT_DTYPE_F32, cond_kernel_cpu);
REGISTER_CPU_KERNEL(cond, INSIGHT_DTYPE_F64, cond_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
