// backends/cpu/kernels/linalg/matmul.cpp
/**
 * @file matmul.cpp
 * @brief CPU kernel for matrix multiplication.
 *
 * Supports vector-vector, matrix-vector, matrix-matrix, and batched matmul.
 * Uses OpenBLAS if available, otherwise falls back to simple loops.
 */

#include "common.h"
#include "../common/half_utils.h"

#include <cstring>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// fallback implementations (when no BLAS acceleration)
// -----------------------------------------------------------------------------

static void matmul_f32_scalar_fallback(const float *a, const float *b, float *c,
                                       int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < k; ++l) {
        sum += a[i * k + l] * b[l * n + j];
      }
      c[i * n + j] = sum;
    }
  }
}

static void matmul_f64_scalar_fallback(const double *a, const double *b,
                                       double *c, int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int l = 0; l < k; ++l) {
        sum += a[i * k + l] * b[l * n + j];
      }
      c[i * n + j] = sum;
    }
  }
}

static void matmul_f32_vector_fallback(const float *a, const float *b, float *c,
                                       int m, int n, int is_a_vec) {
  if (is_a_vec) {
    // vector * matrix: a is 1 x m, b is m x n
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int i = 0; i < m; ++i) {
        sum += a[i] * b[i * n + j];
      }
      c[j] = sum;
    }
  } else {
    // matrix * vector: a is m x n, b is n x 1
    for (int i = 0; i < m; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < n; ++j) {
        sum += a[i * n + j] * b[j];
      }
      c[i] = sum;
    }
  }
}

static void matmul_f64_vector_fallback(const double *a, const double *b,
                                       double *c, int m, int n, int is_a_vec) {
  if (is_a_vec) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int i = 0; i < m; ++i) {
        sum += a[i] * b[i * n + j];
      }
      c[j] = sum;
    }
  } else {
    for (int i = 0; i < m; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += a[i * n + j] * b[j];
      }
      c[i] = sum;
    }
  }
}

// -----------------------------------------------------------------------------
// matrix * matrix (2D)
// -----------------------------------------------------------------------------
static void matmul_f32_matrix_matrix(const float *a, const float *b, float *c,
                                     int m, int k, int n) {
#if HAVE_BLAS_ACCEL
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b,
              n, 0.0f, c, n);
#else
  matmul_f32_scalar_fallback(a, b, c, m, k, n);
#endif
}

static void matmul_f64_matrix_matrix(const double *a, const double *b,
                                     double *c, int m, int k, int n) {
#if HAVE_BLAS_ACCEL
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b,
              n, 0.0, c, n);
#else
  matmul_f64_scalar_fallback(a, b, c, m, k, n);
#endif
}

// -----------------------------------------------------------------------------
// matrix * vector or vector * matrix
// -----------------------------------------------------------------------------
static void matmul_f32_matrix_vector(const float *a, const float *b, float *c,
                                     int m, int n, int is_a_vec) {
#if HAVE_BLAS_ACCEL
  if (is_a_vec) {
    cblas_sgemv(CblasRowMajor, CblasTrans, m, n, 1.0f, a, n, b, 1, 0.0f, c, 1);
  } else {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, a, n, b, 1, 0.0f, c,
                1);
  }
#else
  matmul_f32_vector_fallback(a, b, c, m, n, is_a_vec);
#endif
}

static void matmul_f64_matrix_vector(const double *a, const double *b,
                                     double *c, int m, int n, int is_a_vec) {
#if HAVE_BLAS_ACCEL
  if (is_a_vec) {
    cblas_dgemv(CblasRowMajor, CblasTrans, m, n, 1.0, a, n, b, 1, 0.0, c, 1);
  } else {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, a, n, b, 1, 0.0, c, 1);
  }
#else
  matmul_f64_vector_fallback(a, b, c, m, n, is_a_vec);
#endif
}

// -----------------------------------------------------------------------------
// vector * vector (dot product)
// -----------------------------------------------------------------------------
static float matmul_f32_vector_vector(const float *a, const float *b, int n) {
#if HAVE_BLAS_ACCEL
  return cblas_sdot(n, a, 1, b, 1);
#else
  float sum = 0.0f;
  for (int i = 0; i < n; ++i)
    sum += a[i] * b[i];
  return sum;
#endif
}

static double matmul_f64_vector_vector(const double *a, const double *b,
                                       int n) {
#if HAVE_BLAS_ACCEL
  return cblas_ddot(n, a, 1, b, 1);
#else
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += a[i] * b[i];
  return sum;
#endif
}

// -----------------------------------------------------------------------------
// batched matrix multiplication (ND -> flatten)
// -----------------------------------------------------------------------------
static void matmul_f32_batched(const float *a, const float *b, float *c,
                               int64_t batch, int m, int k, int n) {
  for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
    const float *a_batch = a + b_idx * m * k;
    const float *b_batch = b + b_idx * k * n;
    float *c_batch = c + b_idx * m * n;
    matmul_f32_matrix_matrix(a_batch, b_batch, c_batch, m, k, n);
  }
}

static void matmul_f64_batched(const double *a, const double *b, double *c,
                               int64_t batch, int m, int k, int n) {
  for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
    const double *a_batch = a + b_idx * m * k;
    const double *b_batch = b + b_idx * k * n;
    double *c_batch = c + b_idx * m * n;
    matmul_f64_matrix_matrix(a_batch, b_batch, c_batch, m, k, n);
  }
}

// -----------------------------------------------------------------------------
// half-precision helper: convert -> f32 compute -> convert back
// -----------------------------------------------------------------------------
static C_Status matmul_half_kernel(InsightArray *a, InsightArray *b,
                                  InsightArray *out, bool is_f16) {
  auto to_f32_vec = [&](const uint16_t *src, int64_t n) {
    std::vector<float> dst(n);
    if (is_f16) {
      for (int64_t i = 0; i < n; ++i)
        dst[i] = insight::f16_to_f32(src[i]);
    } else {
      for (int64_t i = 0; i < n; ++i)
        dst[i] = insight::bf16_to_f32(src[i]);
    }
    return dst;
  };

  auto from_f32 = [&](float val) -> uint16_t {
    return is_f16 ? insight::f32_to_f16(val) : insight::f32_to_bf16(val);
  };

  int ndim_a = a->ndim;
  int ndim_b = b->ndim;

  // vector * vector -> scalar
  if (ndim_a == 1 && ndim_b == 1) {
    int n = (int)a->numel;
    auto af = to_f32_vec((const uint16_t *)a->data, n);
    auto bf = to_f32_vec((const uint16_t *)b->data, n);
    float c = matmul_f32_vector_vector(af.data(), bf.data(), n);
    *(uint16_t *)out->data = from_f32(c);
    return C_SUCCESS;
  }

  // matrix * vector
  if (ndim_a == 2 && ndim_b == 1) {
    int m = (int)a->dims[0], n = (int)a->dims[1];
    auto af = to_f32_vec((const uint16_t *)a->data, (int64_t)m * n);
    auto bf = to_f32_vec((const uint16_t *)b->data, n);
    std::vector<float> cf(m);
    matmul_f32_matrix_vector(af.data(), bf.data(), cf.data(), m, n, 0);
    uint16_t *dst = (uint16_t *)out->data;
    for (int i = 0; i < m; ++i)
      dst[i] = from_f32(cf[i]);
    return C_SUCCESS;
  }

  // vector * matrix
  if (ndim_a == 1 && ndim_b == 2) {
    int m = (int)b->dims[0], n = (int)b->dims[1];
    auto af = to_f32_vec((const uint16_t *)a->data, m);
    auto bf = to_f32_vec((const uint16_t *)b->data, (int64_t)m * n);
    std::vector<float> cf(n);
    matmul_f32_matrix_vector(bf.data(), af.data(), cf.data(), m, n, 1);
    uint16_t *dst = (uint16_t *)out->data;
    for (int i = 0; i < n; ++i)
      dst[i] = from_f32(cf[i]);
    return C_SUCCESS;
  }

  // matrix * matrix
  if (ndim_a == 2 && ndim_b == 2) {
    int m = (int)a->dims[0], k = (int)a->dims[1], n = (int)b->dims[1];
    auto af = to_f32_vec((const uint16_t *)a->data, (int64_t)m * k);
    auto bf = to_f32_vec((const uint16_t *)b->data, (int64_t)k * n);
    std::vector<float> cf((int64_t)m * n);
    matmul_f32_matrix_matrix(af.data(), bf.data(), cf.data(), m, k, n);
    uint16_t *dst = (uint16_t *)out->data;
    for (int64_t i = 0; i < (int64_t)m * n; ++i)
      dst[i] = from_f32(cf[i]);
    return C_SUCCESS;
  }

  // batched
  int64_t batch = 1;
  int batch_dims = out->ndim - 2;
  for (int i = 0; i < batch_dims; ++i)
    batch *= out->dims[i];
  int m = (int)out->dims[batch_dims];
  int n = (int)out->dims[batch_dims + 1];
  int k = (ndim_a == 1) ? 1 : (int)a->dims[ndim_a - 1];

  int64_t a_total = batch * m * k;
  int64_t b_total = batch * k * n;
  int64_t c_total = batch * m * n;
  auto af = to_f32_vec((const uint16_t *)a->data, a_total);
  auto bf = to_f32_vec((const uint16_t *)b->data, b_total);
  std::vector<float> cf(c_total);
  matmul_f32_batched(af.data(), bf.data(), cf.data(), batch, m, k, n);
  uint16_t *dst = (uint16_t *)out->data;
  for (int64_t i = 0; i < c_total; ++i)
    dst[i] = from_f32(cf[i]);
  return C_SUCCESS;
}

// -----------------------------------------------------------------------------
// kernel entry point
// -----------------------------------------------------------------------------
C_Status matmul_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *a = (InsightArray *)inputs[1];
  InsightArray *b = (InsightArray *)inputs[2];

  if (!out || !a || !b) {
    cpu_set_last_error("matmul: null array pointer");
    return C_FAILED;
  }

  // Ensure all arrays are contiguous
  if (!cpu_ensure_contiguous(a) || !cpu_ensure_contiguous(b) ||
      !cpu_ensure_contiguous(out))
    return C_FAILED;

  // Handle half-precision via convert->f32->convert-back
  if (out->dtype == INSIGHT_DTYPE_F16 || out->dtype == INSIGHT_DTYPE_BF16) {
    return matmul_half_kernel(a, b, out, out->dtype == INSIGHT_DTYPE_F16);
  }

  int ndim_a = a->ndim;
  int ndim_b = b->ndim;

  // case: vector * vector -> scalar
  if (ndim_a == 1 && ndim_b == 1) {
    int n = (int)a->numel;
    if (b->numel != n) {
      cpu_set_last_error("matmul: vector length mismatch");
      return C_FAILED;
    }
    if (out->dtype == INSIGHT_DTYPE_F32) {
      float *c = (float *)out->data;
      *c = matmul_f32_vector_vector((float *)a->data, (float *)b->data, n);
    } else {
      double *c = (double *)out->data;
      *c = matmul_f64_vector_vector((double *)a->data, (double *)b->data, n);
    }
    return C_SUCCESS;
  }

  // case: matrix * vector
  if (ndim_a == 2 && ndim_b == 1) {
    int m = (int)a->dims[0];
    int n = (int)a->dims[1];
    if (b->numel != n) {
      cpu_set_last_error("matmul: shape mismatch");
      return C_FAILED;
    }
    if (out->dtype == INSIGHT_DTYPE_F32) {
      matmul_f32_matrix_vector((float *)a->data, (float *)b->data,
                               (float *)out->data, m, n, 0);
    } else {
      matmul_f64_matrix_vector((double *)a->data, (double *)b->data,
                               (double *)out->data, m, n, 0);
    }
    return C_SUCCESS;
  }

  // case: vector * matrix
  if (ndim_a == 1 && ndim_b == 2) {
    int m = (int)b->dims[0];
    int n = (int)b->dims[1];
    if (a->numel != m) {
      cpu_set_last_error("matmul: shape mismatch");
      return C_FAILED;
    }
    if (out->dtype == INSIGHT_DTYPE_F32) {
      matmul_f32_matrix_vector((float *)b->data, (float *)a->data,
                               (float *)out->data, m, n, 1);
    } else {
      matmul_f64_matrix_vector((double *)b->data, (double *)a->data,
                               (double *)out->data, m, n, 1);
    }
    return C_SUCCESS;
  }

  // case: matrix * matrix (2D)
  if (ndim_a == 2 && ndim_b == 2) {
    int m = (int)a->dims[0];
    int k = (int)a->dims[1];
    int n = (int)b->dims[1];
    if (k != (int)b->dims[0]) {
      cpu_set_last_error("matmul: shape mismatch");
      return C_FAILED;
    }
    if (out->dtype == INSIGHT_DTYPE_F32) {
      matmul_f32_matrix_matrix((float *)a->data, (float *)b->data,
                               (float *)out->data, m, k, n);
    } else {
      matmul_f64_matrix_matrix((double *)a->data, (double *)b->data,
                               (double *)out->data, m, k, n);
    }
    return C_SUCCESS;
  }

  // case: batched matmul (ND)
  // At this point, we have broadcasted shapes; we can flatten the batch
  // dimensions.
  int64_t batch = 1;
  int batch_dims = out->ndim - 2;
  for (int i = 0; i < batch_dims; ++i) {
    batch *= out->dims[i];
  }
  int m = (int)out->dims[batch_dims];
  int n = (int)out->dims[batch_dims + 1];
  int k_a = (ndim_a == 1) ? 1 : (int)a->dims[ndim_a - 1];
  int k_b = (ndim_b == 1) ? 1 : (int)b->dims[ndim_b - 2];
  int k = k_a; // should equal k_b

  if (out->dtype == INSIGHT_DTYPE_F32) {
    matmul_f32_batched((float *)a->data, (float *)b->data, (float *)out->data,
                       batch, m, k, n);
  } else {
    matmul_f64_batched((double *)a->data, (double *)b->data,
                       (double *)out->data, batch, m, k, n);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(matmul, INSIGHT_DTYPE_F32, matmul_kernel_cpu);
REGISTER_CPU_KERNEL(matmul, INSIGHT_DTYPE_F64, matmul_kernel_cpu);
REGISTER_CPU_KERNEL(matmul, INSIGHT_DTYPE_F16, matmul_kernel_cpu);
REGISTER_CPU_KERNEL(matmul, INSIGHT_DTYPE_BF16, matmul_kernel_cpu);
