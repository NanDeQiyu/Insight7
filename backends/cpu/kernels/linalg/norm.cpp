// backends/cpu/kernels/linalg/norm.cpp
/**
 * @file norm.cpp
 * @brief CPU kernel for vector and matrix norms.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include "../common/half_utils.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

static float norm_vec_f32(const float *data, int n, double ord) {
  if (ord == 2.0) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      sum += data[i] * data[i];
    }
    return sqrtf(sum);
  } else if (ord == 1.0) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      sum += fabsf(data[i]);
    }
    return sum;
  } else if (ord == INFINITY) {
    float max_val = 0.0f;
    for (int i = 0; i < n; ++i) {
      float a = fabsf(data[i]);
      if (a > max_val) {
        max_val = a;
      }
    }
    return max_val;
  } else {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      sum += powf(fabsf(data[i]), (float)ord);
    }
    return powf(sum, 1.0f / (float)ord);
  }
}

static double norm_vec_f64(const double *data, int n, double ord) {
  if (ord == 2.0) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      sum += data[i] * data[i];
    }
    return sqrt(sum);
  } else if (ord == 1.0) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      sum += fabs(data[i]);
    }
    return sum;
  } else if (ord == INFINITY) {
    double max_val = 0.0;
    for (int i = 0; i < n; ++i) {
      double a = fabs(data[i]);
      if (a > max_val) {
        max_val = a;
      }
    }
    return max_val;
  } else {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      sum += pow(fabs(data[i]), ord);
    }
    return pow(sum, 1.0 / ord);
  }
}

static double norm_mat_f64(const double *data, int m, int n, double ord) {
  double *a = (double *)malloc(m * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("norm: memory allocation failed");
    return 0.0;
  }
  cpu_rowmajor_to_colmajor_f64(data, a, m, n);

  double result = 0.0;

  if (ord == 1.0) {
    result = LAPACKE_dlange(LAPACK_COL_MAJOR, '1', m, n, a, m);
  } else if (ord == INFINITY) {
    result = LAPACKE_dlange(LAPACK_COL_MAJOR, 'I', m, n, a, m);
  } else {
    // Frobenius norm
    result = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', m, n, a, m);
  }

  free(a);
  return result;
}

static float norm_mat_f32(const float *data, int m, int n, double ord) {
  float *a = (float *)malloc(m * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("norm: memory allocation failed");
    return 0.0f;
  }
  cpu_rowmajor_to_colmajor_f32(data, a, m, n);

  float result = 0.0f;

  if (ord == 1.0) {
    result = LAPACKE_slange(LAPACK_COL_MAJOR, '1', m, n, a, m);
  } else if (ord == INFINITY) {
    result = LAPACKE_slange(LAPACK_COL_MAJOR, 'I', m, n, a, m);
  } else {
    // Frobenius norm
    result = LAPACKE_slange(LAPACK_COL_MAJOR, 'F', m, n, a, m);
  }

  free(a);
  return result;
}

C_Status norm_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  double ord = *(double *)inputs[2];

  if (!out || !x) {
    cpu_set_last_error("norm: null array pointer");
    return C_FAILED;
  }

  if (!cpu_ensure_contiguous(x)) {
    return C_FAILED;
  }

  cpu_set_last_error("");

  if (x->ndim == 1) {
    int n = (int)x->numel;
    if (x->dtype == INSIGHT_DTYPE_F32) {
      float *dst = (float *)out->data;
      *dst = norm_vec_f32((float *)x->data, n, ord);
    } else if (x->dtype == INSIGHT_DTYPE_F16 ||
               x->dtype == INSIGHT_DTYPE_BF16) {
      const uint16_t *src = (const uint16_t *)x->data;
      uint16_t *dst = (uint16_t *)out->data;
      std::vector<float> tmp(n);
      if (x->dtype == INSIGHT_DTYPE_F16) {
        for (int i = 0; i < n; ++i)
          tmp[i] = insight::f16_to_f32(src[i]);
      } else {
        for (int i = 0; i < n; ++i)
          tmp[i] = insight::bf16_to_f32(src[i]);
      }
      float result = norm_vec_f32(tmp.data(), n, ord);
      *dst = (x->dtype == INSIGHT_DTYPE_F16) ? insight::f32_to_f16(result)
                                              : insight::f32_to_bf16(result);
    } else {
      double *dst = (double *)out->data;
      *dst = norm_vec_f64((double *)x->data, n, ord);
    }
  } else if (x->ndim == 2) {
    int m = (int)x->dims[0];
    int n = (int)x->dims[1];
    if (x->dtype == INSIGHT_DTYPE_F32) {
      float *dst = (float *)out->data;
      *dst = norm_mat_f32((float *)x->data, m, n, ord);
    } else if (x->dtype == INSIGHT_DTYPE_F16 ||
               x->dtype == INSIGHT_DTYPE_BF16) {
      const uint16_t *src = (const uint16_t *)x->data;
      uint16_t *dst = (uint16_t *)out->data;
      int64_t total = (int64_t)m * n;
      std::vector<float> tmp(total);
      if (x->dtype == INSIGHT_DTYPE_F16) {
        for (int64_t i = 0; i < total; ++i)
          tmp[i] = insight::f16_to_f32(src[i]);
      } else {
        for (int64_t i = 0; i < total; ++i)
          tmp[i] = insight::bf16_to_f32(src[i]);
      }
      float result = norm_mat_f32(tmp.data(), m, n, ord);
      *dst = (x->dtype == INSIGHT_DTYPE_F16) ? insight::f32_to_f16(result)
                                              : insight::f32_to_bf16(result);
    } else {
      double *dst = (double *)out->data;
      *dst = norm_mat_f64((double *)x->data, m, n, ord);
    }
  } else {
    cpu_set_last_error("norm: unsupported dimension");
    return C_FAILED;
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(norm, INSIGHT_DTYPE_F32, norm_kernel_cpu);
REGISTER_CPU_KERNEL(norm, INSIGHT_DTYPE_F64, norm_kernel_cpu);
REGISTER_CPU_KERNEL(norm, INSIGHT_DTYPE_F16, norm_kernel_cpu);
REGISTER_CPU_KERNEL(norm, INSIGHT_DTYPE_BF16, norm_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status norm_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("norm: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(norm, INSIGHT_DTYPE_F32, norm_kernel_cpu);
REGISTER_CPU_KERNEL(norm, INSIGHT_DTYPE_F64, norm_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
