// backends/cpu/kernels/linalg/eigvalsh.cpp
/**
 * @file eigvalsh.cpp
 * @brief CPU kernel for eigenvalues of symmetric matrix.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <cstdlib>
#include <cstring>

static void eigvalsh_f32(const float *src, float *vals, int n, int uplo) {
  // Copy to column major order
  float *a = (float *)malloc(n * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("eigvalsh: memory allocation failed");
    return;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * n] = src[i * n + j];
    }
  }

  char uplo_char = uplo ? 'U' : 'L';
  char jobz = 'N'; // Only eigenvalues ​​are calculated, eigenvectors are not calculated

  // LAPACKE automatically manages workspace internally
  int info = LAPACKE_ssyev(LAPACK_COL_MAJOR, jobz, uplo_char, n, a, n, vals);

  if (info != 0) {
    cpu_set_last_error("eigvalsh: LAPACK ssyev failed");
  }

  free(a);
}

static void eigvalsh_f64(const double *src, double *vals, int n, int uplo) {
  double *a = (double *)malloc(n * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("eigvalsh: memory allocation failed");
    return;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * n] = src[i * n + j];
    }
  }

  char uplo_char = uplo ? 'U' : 'L';
  char jobz = 'N';

  int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, jobz, uplo_char, n, a, n, vals);

  if (info != 0) {
    cpu_set_last_error("eigvalsh: LAPACK dsyev failed");
  }

  free(a);
}

C_Status eigvalsh_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  int uplo = *(int *)inputs[2];

  if (!out || !x) {
    cpu_set_last_error("eigvalsh: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(out)) {
    return C_FAILED;
  }

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("eigvalsh: matrix must be square");
    return C_FAILED;
  }

  cpu_set_last_error("");

  if (x->dtype == INSIGHT_DTYPE_F32) {
    eigvalsh_f32((float *)x->data, (float *)out->data, n, uplo);
  } else {
    eigvalsh_f64((double *)x->data, (double *)out->data, n, uplo);
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(eigvalsh, INSIGHT_DTYPE_F32, eigvalsh_kernel_cpu);
REGISTER_CPU_KERNEL(eigvalsh, INSIGHT_DTYPE_F64, eigvalsh_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status eigvalsh_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("eigvalsh: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(eigvalsh, INSIGHT_DTYPE_F32, eigvalsh_kernel_cpu);
REGISTER_CPU_KERNEL(eigvalsh, INSIGHT_DTYPE_F64, eigvalsh_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
