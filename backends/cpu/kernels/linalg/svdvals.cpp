// backends/cpu/kernels/linalg/svdvals.cpp
/**
 * @file svdvals.cpp
 * @brief CPU kernel for singular values only.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <cstdlib>
#include <cstring>

static void svdvals_f32(const float *src, float *s, int m, int n) {
  float *a = (float *)malloc(m * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("svdvals: memory allocation failed");
    return;
  }
  memcpy(a, src, m * n * sizeof(float));

  int min_mn = m < n ? m : n;
  char jobu = 'N';
  char jobvt = 'N';
  int ldu = m;
  int ldvt = n;

  float *superb = (float *)malloc((min_mn - 1) * sizeof(float));
  if (!superb) {
    cpu_set_last_error("svdvals: memory allocation failed");
    free(a);
    return;
  }

  int info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, a, m, s, NULL,
                            ldu, NULL, ldvt, superb);

  free(a);
  free(superb);

  if (info != 0) {
    cpu_set_last_error("svdvals: LAPACK sgesvd failed");
  }
}

static void svdvals_f64(const double *src, double *s, int m, int n) {
  double *a = (double *)malloc(m * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("svdvals: memory allocation failed");
    return;
  }
  memcpy(a, src, m * n * sizeof(double));

  int min_mn = m < n ? m : n;
  char jobu = 'N';
  char jobvt = 'N';
  int ldu = m;
  int ldvt = n;

  double *superb = (double *)malloc((min_mn - 1) * sizeof(double));
  if (!superb) {
    cpu_set_last_error("svdvals: memory allocation failed");
    free(a);
    return;
  }

  int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, a, m, s, NULL,
                            ldu, NULL, ldvt, superb);

  free(a);
  free(superb);

  if (info != 0) {
    cpu_set_last_error("svdvals: LAPACK dgesvd failed");
  }
}

C_Status svdvals_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("svdvals: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(out)) {
    return C_FAILED;
  }

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  cpu_set_last_error("");

  if (x->dtype == INSIGHT_DTYPE_F32) {
    svdvals_f32((float *)x->data, (float *)out->data, m, n);
  } else {
    svdvals_f64((double *)x->data, (double *)out->data, m, n);
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(svdvals, INSIGHT_DTYPE_F32, svdvals_kernel_cpu);
REGISTER_CPU_KERNEL(svdvals, INSIGHT_DTYPE_F64, svdvals_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status svdvals_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("svdvals: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(svdvals, INSIGHT_DTYPE_F32, svdvals_kernel_cpu);
REGISTER_CPU_KERNEL(svdvals, INSIGHT_DTYPE_F64, svdvals_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
