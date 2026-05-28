// backends/cpu/kernels/linalg/svd.cpp
/**
 * @file svd.cpp
 * @brief CPU kernel for Singular Value Decomposition using LAPACK.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <cstdlib>
#include <cstring>

static void svd_f32(const float *src, float *u, float *s, float *vt, int m,
                    int n, int full) {

  int min_mn = m < n ? m : n;

  // Define variables
  char jobu = full ? 'A' : 'S';
  char jobvt = full ? 'A' : 'S';
  int ldu = m;
  int ldvt = full ? n : min_mn;

  float *a = (float *)malloc(m * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("svd: memory allocation failed");
    return;
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * m] = src[i * n + j];
    }
  }

  float *superb = NULL;
  if (min_mn > 1) {
    superb = (float *)malloc((min_mn - 1) * sizeof(float));
    if (!superb) {
      cpu_set_last_error("svd: memory allocation failed");
      free(a);
      return;
    }
  }

  int info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, a, m, s, u,
                            ldu, vt, ldvt, superb);
  if (info != 0) {
    cpu_set_last_error("svd: LAPACK sgesvd failed");
  }

  free(a);
  if (superb) {
    free(superb);
  }
}

static void svd_f64(const double *src, double *u, double *s, double *vt, int m,
                    int n, int full) {

  int min_mn = m < n ? m : n;
  char jobu = full ? 'A' : 'S';
  char jobvt = full ? 'A' : 'S';
  int ldu = m;
  int ldvt = full ? n : min_mn;

  double *a = (double *)malloc(m * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("svd: memory allocation failed");
    return;
  }
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * m] = src[i * n + j];
    }
  }

  double *superb = NULL;
  if (min_mn > 1) {
    superb = (double *)malloc((min_mn - 1) * sizeof(double));
    if (!superb) {
      cpu_set_last_error("svd: memory allocation failed");
      free(a);
      return;
    }
  }

  int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, a, m, s, u,
                            ldu, vt, ldvt, superb);

  if (info != 0) {
    cpu_set_last_error("svd: LAPACK dgesvd failed");
  }

  free(a);
  if (superb)
    free(superb);
}

C_Status svd_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *U = (InsightArray *)outputs[0];
  InsightArray *S = (InsightArray *)outputs[1];
  InsightArray *VT = (InsightArray *)outputs[2];
  InsightArray *x = (InsightArray *)inputs[3];
  int full = *(int *)inputs[4];

  if (!U || !S || !VT || !x) {
    cpu_set_last_error("svd: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(U) ||
      !cpu_ensure_contiguous(S) || !cpu_ensure_contiguous(VT)) {
    return C_FAILED;
  }

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  cpu_set_last_error("");

  if (x->dtype == INSIGHT_DTYPE_F32) {
    svd_f32((float *)x->data, (float *)U->data, (float *)S->data,
            (float *)VT->data, m, n, full);
  } else {
    svd_f64((double *)x->data, (double *)U->data, (double *)S->data,
            (double *)VT->data, m, n, full);
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_F32, svd_kernel_cpu);
REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_F64, svd_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status svd_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("svd: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_F32, svd_kernel_cpu);
REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_F64, svd_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
