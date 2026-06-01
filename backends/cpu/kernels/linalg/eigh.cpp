// backends/cpu/kernels/linalg/eigh.cpp
/**
 * @file eigh.cpp
 * @brief CPU kernel for symmetric/Hermitian eigenvalue decomposition.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "../common/half_utils.h"
#include "common.h"
#include <cstdlib>
#include <cstring>
#include <vector>

static void eigh_f32(const float *src, float *vals, float *vecs, int n,
                     int uplo) {
  // Copy to column major order
  float *a = (float *)malloc(n * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("eigh: memory allocation failed");
    return;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * n] = src[i * n + j];
    }
  }

  char uplo_char = uplo ? 'U' : 'L';
  char jobz = 'V'; // Calculate eigenvalues ​​and eigenvectors

  // LAPACKE automatically manages workspace internally
  int info = LAPACKE_ssyev(LAPACK_COL_MAJOR, jobz, uplo_char, n, a, n, vals);

  if (info != 0) {
    cpu_set_last_error("eigh: LAPACK ssyev failed");
    free(a);
    return;
  }

  // Output feature vector (convert from column major order to row major order)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      vecs[i * n + j] = a[i + j * n];
    }
  }

  free(a);
}

static void eigh_f64(const double *src, double *vals, double *vecs, int n,
                     int uplo) {
  double *a = (double *)malloc(n * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("eigh: memory allocation failed");
    return;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * n] = src[i * n + j];
    }
  }

  char uplo_char = uplo ? 'U' : 'L';
  char jobz = 'V';

  int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, jobz, uplo_char, n, a, n, vals);

  if (info != 0) {
    cpu_set_last_error("eigh: LAPACK dsyev failed");
    free(a);
    return;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      vecs[i * n + j] = a[i + j * n];
    }
  }

  free(a);
}

C_Status eigh_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *vals = (InsightArray *)outputs[0];
  InsightArray *vecs = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];
  int uplo = *(int *)inputs[3];

  if (!vals || !vecs || !x) {
    cpu_set_last_error("eigh: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(vals) ||
      !cpu_ensure_contiguous(vecs)) {
    return C_FAILED;
  }

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("eigh: matrix must be square");
    return C_FAILED;
  }

  cpu_set_last_error("");

  if (x->dtype == INSIGHT_DTYPE_F32) {
    eigh_f32((float *)x->data, (float *)vals->data, (float *)vecs->data, n,
             uplo);
  } else if (x->dtype == INSIGHT_DTYPE_F16 || x->dtype == INSIGHT_DTYPE_BF16) {
    const uint16_t *x_src = (const uint16_t *)x->data;
    uint16_t *vals_dst = (uint16_t *)vals->data;
    uint16_t *vecs_dst = (uint16_t *)vecs->data;
    bool is_f16 = (x->dtype == INSIGHT_DTYPE_F16);
    int64_t total = (int64_t)n * n;
    std::vector<float> x_f32(total), vals_f32(n), vecs_f32(total);
    if (is_f16) {
      for (int64_t i = 0; i < total; ++i)
        x_f32[i] = insight::f16_to_f32(x_src[i]);
    } else {
      for (int64_t i = 0; i < total; ++i)
        x_f32[i] = insight::bf16_to_f32(x_src[i]);
    }
    eigh_f32(x_f32.data(), vals_f32.data(), vecs_f32.data(), n, uplo);
    if (is_f16) {
      for (int i = 0; i < n; ++i)
        vals_dst[i] = insight::f32_to_f16(vals_f32[i]);
      for (int64_t i = 0; i < total; ++i)
        vecs_dst[i] = insight::f32_to_f16(vecs_f32[i]);
    } else {
      for (int i = 0; i < n; ++i)
        vals_dst[i] = insight::f32_to_bf16(vals_f32[i]);
      for (int64_t i = 0; i < total; ++i)
        vecs_dst[i] = insight::f32_to_bf16(vecs_f32[i]);
    }
  } else {
    eigh_f64((double *)x->data, (double *)vals->data, (double *)vecs->data, n,
             uplo);
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(eigh, INSIGHT_DTYPE_F32, eigh_kernel_cpu);
REGISTER_CPU_KERNEL(eigh, INSIGHT_DTYPE_F64, eigh_kernel_cpu);
REGISTER_CPU_KERNEL(eigh, INSIGHT_DTYPE_F16, eigh_kernel_cpu);
REGISTER_CPU_KERNEL(eigh, INSIGHT_DTYPE_BF16, eigh_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status eigh_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("eigh: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(eigh, INSIGHT_DTYPE_F32, eigh_kernel_cpu);
REGISTER_CPU_KERNEL(eigh, INSIGHT_DTYPE_F64, eigh_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
