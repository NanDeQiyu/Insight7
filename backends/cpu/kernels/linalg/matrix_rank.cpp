// backends/cpu/kernels/linalg/matrix_rank.cpp
/**
 * @file matrix_rank.cpp
 * @brief CPU kernel for matrix rank using SVD.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <cstring>

static int rank_f64(const double *src, int m, int n, double tol) {
  int min_mn = m < n ? m : n;
  double *a = (double *)malloc(m * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("matrix_rank: memory allocation failed");
    return 0;
  }
  // row major → column major (LAPACK required)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * m] = src[i * n + j];
    }
  }

  double *s = (double *)malloc(min_mn * sizeof(double));
  if (!s) {
    cpu_set_last_error("matrix_rank: memory allocation failed");
    free(a);
    return 0;
  }

  double *superb = (double *)malloc((min_mn - 1) * sizeof(double));
  if (!superb) {
    cpu_set_last_error("matrix_rank: memory allocation failed");
    free(a);
    free(s);
    return 0;
  }

  // Compute singular values ​​using LAPACKE_dgesvd
  int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N', m, n, a, m, s, NULL, m,
                            NULL, n, superb);

  if (info != 0) {
    cpu_set_last_error("matrix_rank: LAPACK dgesvd failed");
    free(a);
    free(s);
    free(superb);
    return 0;
  }

  double max_s = s[0];
  double actual_tol;
  if (tol < 0) {
    // Use LAPACK recommended tolerance formulas
    actual_tol = max_s * std::max(m, n) * DBL_EPSILON;
  } else {
    actual_tol = tol;
  }

  int rank = 0;
  for (int i = 0; i < min_mn; ++i) {
    if (s[i] > actual_tol) {
      ++rank;
    }
  }

  free(a);
  free(s);
  free(superb);
  return rank;
}

static int rank_f32(const float *src, int m, int n, double tol) {
  int min_mn = m < n ? m : n;
  float *a = (float *)malloc(m * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("matrix_rank: memory allocation failed");
    return 0;
  }
  // row major → column major (LAPACK required)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * m] = src[i * n + j];
    }
  }

  float *s = (float *)malloc(min_mn * sizeof(float));
  if (!s) {
    cpu_set_last_error("matrix_rank: memory allocation failed");
    free(a);
    return 0;
  }

  float *superb = (float *)malloc((min_mn - 1) * sizeof(float));
  if (!superb) {
    cpu_set_last_error("matrix_rank: memory allocation failed");
    free(a);
    free(s);
    return 0;
  }

  // Compute singular values ​​using LAPACKE_sgesvd
  int info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'N', 'N', m, n, a, m, s, NULL, m,
                            NULL, n, superb);

  if (info != 0) {
    cpu_set_last_error("matrix_rank: LAPACK sgesvd failed");
    free(a);
    free(s);
    free(superb);
    return 0;
  }

  float max_s = s[0];
  float actual_tol;
  if (tol < 0) {
    // Use LAPACK recommended tolerance formulas
    actual_tol = max_s * std::max(m, n) * FLT_EPSILON;
  } else {
    actual_tol = (float)tol;
  }

  int rank = 0;
  for (int i = 0; i < min_mn; ++i) {
    if (s[i] > actual_tol) {
      ++rank;
    }
  }

  free(a);
  free(s);
  free(superb);
  return rank;
}

C_Status matrix_rank_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  double tol = *(double *)inputs[2];

  if (!out || !x) {
    cpu_set_last_error("matrix_rank: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(x)) {
    return C_FAILED;
  }

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  cpu_set_last_error("");

  int r;
  if (x->dtype == INSIGHT_DTYPE_F32) {
    r = rank_f32((float *)x->data, m, n, tol);
  } else {
    r = rank_f64((double *)x->data, m, n, tol);
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  *(int64_t *)out->data = r;
  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(matrix_rank, INSIGHT_DTYPE_F32, matrix_rank_kernel_cpu);
REGISTER_CPU_KERNEL(matrix_rank, INSIGHT_DTYPE_F64, matrix_rank_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status matrix_rank_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("matrix_rank: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(matrix_rank, INSIGHT_DTYPE_F32, matrix_rank_kernel_cpu);
REGISTER_CPU_KERNEL(matrix_rank, INSIGHT_DTYPE_F64, matrix_rank_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
