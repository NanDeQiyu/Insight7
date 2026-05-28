// backends/cpu/kernels/linalg/eigh.cpp
/**
 * @file eigh.cpp
 * @brief CPU kernel for symmetric/Hermitian eigenvalue decomposition.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <cstdlib>
#include <cstring>

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
