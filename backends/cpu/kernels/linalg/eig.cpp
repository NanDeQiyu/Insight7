// backends/cpu/kernels/linalg/eig.cpp
/**
 * @file eig.cpp
 * @brief CPU kernel for eigenvalues and eigenvectors (complex).
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <complex>
#include <cstdlib>
#include <cstring>

static void eig_f32(const float *src, float *vals_real, float *vals_imag,
                    float *vecs_real, float *vecs_imag, int n) {
  float *a = (float *)malloc(n * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("eig: memory allocation failed");
    return;
  }
  memcpy(a, src, n * n * sizeof(float));

  float *wr = (float *)malloc(n * sizeof(float));
  float *wi = (float *)malloc(n * sizeof(float));
  float *vl = NULL;
  float *vr = (float *)malloc(n * n * sizeof(float));
  float *work = (float *)malloc(1 * sizeof(float));
  int lwork = -1;
  int info;

  if (!wr || !wi || !vr || !work) {
    cpu_set_last_error("eig: memory allocation failed");
    free(a);
    free(wr);
    free(wi);
    free(vr);
    free(work);
    return;
  }

  // workspace query
  LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', n, a, n, wr, wi, vl, 1, vr, n);

  lwork = (int)work[0];
  float *work_opt = (float *)realloc(work, lwork * sizeof(float));
  if (!work_opt) {
    cpu_set_last_error("eig: memory allocation failed");
    free(a);
    free(wr);
    free(wi);
    free(vr);
    free(work);
    return;
  }
  work = work_opt;

  LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', n, a, n, wr, wi, vl, 1, vr, n);

  // eigenvalues
  for (int i = 0; i < n; ++i) {
    vals_real[i] = wr[i];
    vals_imag[i] = wi[i];
  }

  // eigenvectors (complex: real+imag pairs)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      vecs_real[i * n + j] = vr[j * n + i];
      vecs_imag[i * n + j] = (wi[j] != 0.0f) ? vr[j * n + i + n] : 0.0f;
    }
  }

  free(a);
  free(wr);
  free(wi);
  free(vr);
  free(work);
}

static void eig_f64(const double *src, double *vals_real, double *vals_imag,
                    double *vecs_real, double *vecs_imag, int n) {
  double *a = (double *)malloc(n * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("eig: memory allocation failed");
    return;
  }
  memcpy(a, src, n * n * sizeof(double));

  double *wr = (double *)malloc(n * sizeof(double));
  double *wi = (double *)malloc(n * sizeof(double));
  double *vl = NULL;
  double *vr = (double *)malloc(n * n * sizeof(double));
  double *work = (double *)malloc(1 * sizeof(double));
  int lwork = -1;
  int info;

  if (!wr || !wi || !vr || !work) {
    cpu_set_last_error("eig: memory allocation failed");
    free(a);
    free(wr);
    free(wi);
    free(vr);
    free(work);
    return;
  }

  // workspace query
  LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', n, a, n, wr, wi, vl, 1, vr, n);

  lwork = (int)work[0];
  double *work_opt = (double *)realloc(work, lwork * sizeof(double));
  if (!work_opt) {
    cpu_set_last_error("eig: memory allocation failed");
    free(a);
    free(wr);
    free(wi);
    free(vr);
    free(work);
    return;
  }
  work = work_opt;

  LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', n, a, n, wr, wi, vl, 1, vr, n);

  // eigenvalues
  for (int i = 0; i < n; ++i) {
    vals_real[i] = wr[i];
    vals_imag[i] = wi[i];
  }

  // eigenvectors (complex: real+imag pairs)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      vecs_real[i * n + j] = vr[j * n + i];
      vecs_imag[i * n + j] = (wi[j] != 0.0) ? vr[j * n + i + n] : 0.0;
    }
  }

  free(a);
  free(wr);
  free(wi);
  free(vr);
  free(work);
}

C_Status eig_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *vals_real = (InsightArray *)outputs[0];
  InsightArray *vals_imag = (InsightArray *)outputs[1];
  InsightArray *vecs_real = (InsightArray *)outputs[2];
  InsightArray *vecs_imag = (InsightArray *)outputs[3];
  InsightArray *x = (InsightArray *)inputs[4];

  if (!vals_real || !vals_imag || !vecs_real || !vecs_imag || !x) {
    cpu_set_last_error("eig: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(vals_real) ||
      !cpu_ensure_contiguous(vals_imag) || !cpu_ensure_contiguous(vecs_real) ||
      !cpu_ensure_contiguous(vecs_imag)) {
    return C_FAILED;
  }

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("eig: matrix must be square");
    return C_FAILED;
  }

  cpu_set_last_error("");

  if (x->dtype == INSIGHT_DTYPE_F32) {
    eig_f32((float *)x->data, (float *)vals_real->data,
            (float *)vals_imag->data, (float *)vecs_real->data,
            (float *)vecs_imag->data, n);
  } else {
    eig_f64((double *)x->data, (double *)vals_real->data,
            (double *)vals_imag->data, (double *)vecs_real->data,
            (double *)vecs_imag->data, n);
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(eig, INSIGHT_DTYPE_F32, eig_kernel_cpu);
REGISTER_CPU_KERNEL(eig, INSIGHT_DTYPE_F64, eig_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status eig_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("eig: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(eig, INSIGHT_DTYPE_F32, eig_kernel_cpu);
REGISTER_CPU_KERNEL(eig, INSIGHT_DTYPE_F64, eig_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
