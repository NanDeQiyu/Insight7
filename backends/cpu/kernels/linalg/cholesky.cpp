// backends/cpu/kernels/linalg/cholesky.cpp
/**
 * @file cholesky.cpp
 * @brief CPU kernel for Cholesky decomposition using LAPACK.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "../common/half_utils.h"
#include "common.h"
#include <cstdlib>
#include <cstring>
#include <vector>

static void cholesky_f64(const double *src, double *dst, int n, int lower) {
  double *a = (double *)malloc(n * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("cholesky: memory allocation failed");
    return;
  }
  cpu_rowmajor_to_colmajor_f64(src, a, n, n);

  int matrix_layout = LAPACK_COL_MAJOR;
  char uplo = lower ? 'L' : 'U';
  lapack_int info = LAPACKE_dpotrf(matrix_layout, uplo, n, a, n);

  if (info != 0) {
    cpu_set_last_error("cholesky: matrix not positive definite");
    free(a);
    return;
  }

  // Extract lower triangle or upper triangle
  if (lower) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i >= j) {
          dst[i * n + j] = a[i + j * n];
        } else {
          dst[i * n + j] = 0.0;
        }
      }
    }
  } else {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i <= j) {
          dst[i * n + j] = a[i + j * n];
        } else {
          dst[i * n + j] = 0.0;
        }
      }
    }
  }

  free(a);
}

static void cholesky_f32(const float *src, float *dst, int n, int lower) {
  double *src_f64 = (double *)malloc(n * n * sizeof(double));
  double *dst_f64 = (double *)malloc(n * n * sizeof(double));

  if (!src_f64 || !dst_f64) {
    cpu_set_last_error("cholesky: memory allocation failed");
    free(src_f64);
    free(dst_f64);
    return;
  }

  for (int i = 0; i < n * n; ++i) {
    src_f64[i] = src[i];
  }
  cholesky_f64(src_f64, dst_f64, n, lower);
  if (cpu_get_last_error() && cpu_get_last_error()[0] != '\0') {
    free(src_f64);
    free(dst_f64);
    return;
  }
  for (int i = 0; i < n * n; ++i) {
    dst[i] = (float)dst_f64[i];
  }

  free(src_f64);
  free(dst_f64);
}

C_Status cholesky_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  int lower = *(int *)inputs[2];

  if (!out || !x) {
    cpu_set_last_error("cholesky: null array pointer");
    return C_FAILED;
  }

  // Ensure input and output are continuous
  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(out)) {
    return C_FAILED;
  }

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("cholesky: matrix must be square");
    return C_FAILED;
  }

  // Clear previous error messages
  cpu_set_last_error("");

  if (x->dtype == INSIGHT_DTYPE_F32) {
    cholesky_f32((float *)x->data, (float *)out->data, n, lower);
  } else if (x->dtype == INSIGHT_DTYPE_F16 || x->dtype == INSIGHT_DTYPE_BF16) {
    const uint16_t *x_src = (const uint16_t *)x->data;
    uint16_t *dst = (uint16_t *)out->data;
    bool is_f16 = (x->dtype == INSIGHT_DTYPE_F16);
    int64_t total = (int64_t)n * n;
    std::vector<float> x_f32(total), dst_f32(total);
    if (is_f16) {
      for (int64_t i = 0; i < total; ++i)
        x_f32[i] = insight::f16_to_f32(x_src[i]);
    } else {
      for (int64_t i = 0; i < total; ++i)
        x_f32[i] = insight::bf16_to_f32(x_src[i]);
    }
    cholesky_f32(x_f32.data(), dst_f32.data(), n, lower);
    if (is_f16) {
      for (int64_t i = 0; i < total; ++i)
        dst[i] = insight::f32_to_f16(dst_f32[i]);
    } else {
      for (int64_t i = 0; i < total; ++i)
        dst[i] = insight::f32_to_bf16(dst_f32[i]);
    }
  } else {
    cholesky_f64((double *)x->data, (double *)out->data, n, lower);
  }

  // Check if any errors occurred
  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {

    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(cholesky, INSIGHT_DTYPE_F32, cholesky_kernel_cpu);
REGISTER_CPU_KERNEL(cholesky, INSIGHT_DTYPE_F64, cholesky_kernel_cpu);
REGISTER_CPU_KERNEL(cholesky, INSIGHT_DTYPE_F16, cholesky_kernel_cpu);
REGISTER_CPU_KERNEL(cholesky, INSIGHT_DTYPE_BF16, cholesky_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status cholesky_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("cholesky: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(cholesky, INSIGHT_DTYPE_F32, cholesky_kernel_cpu);
REGISTER_CPU_KERNEL(cholesky, INSIGHT_DTYPE_F64, cholesky_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
