// backends/cpu/kernels/linalg/solve.cpp
/**
 * @file solve.cpp
 * @brief CPU kernel for solving linear systems AX = B using LAPACK.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include "../common/half_utils.h"

#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

static void solve_f32(const float *A, const float *B, float *X, int n,
                      int nrhs) {
  // Convert A from row major to column major (LAPACK expects column major)
  float *a = (float *)malloc(n * n * sizeof(float));
  cpu_rowmajor_to_colmajor_f32(A, a, n, n);

  // Convert B from row major to column major
  float *b = (float *)malloc(n * nrhs * sizeof(float));
  cpu_rowmajor_to_colmajor_f32(B, b, n, nrhs);

  int *ipiv = (int *)malloc(n * sizeof(int));
  int info;
  sgesv_(&n, &nrhs, a, &n, ipiv, b, &n, &info);

  // Convert results from column major back to row major
  cpu_colmajor_to_rowmajor_f32(b, X, n, nrhs);

  free(a);
  free(b);
  free(ipiv);
  if (info != 0) {
    cpu_set_last_error("solve: LAPACK sgesv failed (singular matrix)");
  }
}

static void solve_f64(const double *A, const double *B, double *X, int n,
                      int nrhs) {
  // Convert A from row major to column major (LAPACK expects column major)
  double *a = (double *)malloc(n * n * sizeof(double));
  cpu_rowmajor_to_colmajor_f64(A, a, n, n);

  // Convert B from row major to column major
  double *b = (double *)malloc(n * nrhs * sizeof(double));
  cpu_rowmajor_to_colmajor_f64(B, b, n, nrhs);

  int *ipiv = (int *)malloc(n * sizeof(int));
  int info;
  dgesv_(&n, &nrhs, a, &n, ipiv, b, &n, &info);

  // Convert results from column major back to row major
  cpu_colmajor_to_rowmajor_f64(b, X, n, nrhs);

  free(a);
  free(b);
  free(ipiv);
  if (info != 0) {
    cpu_set_last_error("solve: LAPACK dgesv failed (singular matrix)");
  }
}

C_Status solve_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *A = (InsightArray *)inputs[1];
  InsightArray *B = (InsightArray *)inputs[2];

  if (!out || !A || !B) {
    cpu_set_last_error("solve: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(A) || !cpu_ensure_contiguous(B) ||
      !cpu_ensure_contiguous(out))
    return C_FAILED;

  int n = (int)A->dims[0];
  int nrhs = (B->ndim == 1) ? 1 : (int)B->dims[1];
  if (A->dims[1] != n) {
    cpu_set_last_error("solve: A must be square");
    return C_FAILED;
  }

  if (A->dtype == INSIGHT_DTYPE_F32) {
    solve_f32((float *)A->data, (float *)B->data, (float *)out->data, n, nrhs);
  } else if (A->dtype == INSIGHT_DTYPE_F16 ||
             A->dtype == INSIGHT_DTYPE_BF16) {
    const uint16_t *a_src = (const uint16_t *)A->data;
    const uint16_t *b_src = (const uint16_t *)B->data;
    uint16_t *x_dst = (uint16_t *)out->data;
    int64_t a_total = (int64_t)n * n;
    int64_t b_total = (int64_t)n * nrhs;
    int64_t x_total = (int64_t)n * nrhs;
    std::vector<float> a_f32(a_total), b_f32(b_total), x_f32(x_total);
    bool is_f16 = (A->dtype == INSIGHT_DTYPE_F16);
    if (is_f16) {
      for (int64_t i = 0; i < a_total; ++i)
        a_f32[i] = insight::f16_to_f32(a_src[i]);
      for (int64_t i = 0; i < b_total; ++i)
        b_f32[i] = insight::f16_to_f32(b_src[i]);
    } else {
      for (int64_t i = 0; i < a_total; ++i)
        a_f32[i] = insight::bf16_to_f32(a_src[i]);
      for (int64_t i = 0; i < b_total; ++i)
        b_f32[i] = insight::bf16_to_f32(b_src[i]);
    }
    solve_f32(a_f32.data(), b_f32.data(), x_f32.data(), n, nrhs);
    if (is_f16) {
      for (int64_t i = 0; i < x_total; ++i)
        x_dst[i] = insight::f32_to_f16(x_f32[i]);
    } else {
      for (int64_t i = 0; i < x_total; ++i)
        x_dst[i] = insight::f32_to_bf16(x_f32[i]);
    }
  } else {
    solve_f64((double *)A->data, (double *)B->data, (double *)out->data, n,
              nrhs);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(solve, INSIGHT_DTYPE_F32, solve_kernel_cpu);
REGISTER_CPU_KERNEL(solve, INSIGHT_DTYPE_F64, solve_kernel_cpu);
REGISTER_CPU_KERNEL(solve, INSIGHT_DTYPE_F16, solve_kernel_cpu);
REGISTER_CPU_KERNEL(solve, INSIGHT_DTYPE_BF16, solve_kernel_cpu);

#endif
