// backends/cpu/kernels/linalg/dot.cpp
/**
 * @file dot.cpp
 * @brief CPU kernel for dot product of two 1D vectors.
 */

#include "../common/half_utils.h"
#include "common.h"

#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

static float dot_f32_fallback(const float *a, const float *b, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; ++i)
    sum += a[i] * b[i];
  return sum;
}

static double dot_f64_fallback(const double *a, const double *b, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += a[i] * b[i];
  return sum;
}

C_Status dot_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *a = (InsightArray *)inputs[1];
  InsightArray *b = (InsightArray *)inputs[2];

  if (!out || !a || !b) {
    cpu_set_last_error("dot: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(a) || !cpu_ensure_contiguous(b))
    return C_FAILED;

  int n = (int)a->numel;
  if (b->numel != n) {
    cpu_set_last_error("dot: vector length mismatch");
    return C_FAILED;
  }

  if (out->dtype == INSIGHT_DTYPE_F32) {
    float *c = (float *)out->data;
#if HAVE_BLAS_ACCEL
    *c = cblas_sdot(n, (float *)a->data, 1, (float *)b->data, 1);
#else
    *c = dot_f32_fallback((float *)a->data, (float *)b->data, n);
#endif
  } else if (out->dtype == INSIGHT_DTYPE_F16 ||
             out->dtype == INSIGHT_DTYPE_BF16) {
    const uint16_t *a_data = (const uint16_t *)a->data;
    const uint16_t *b_data = (const uint16_t *)b->data;
    uint16_t *dst = (uint16_t *)out->data;
    float sum = 0.0f;
    if (out->dtype == INSIGHT_DTYPE_F16) {
      for (int i = 0; i < n; ++i)
        sum += insight::f16_to_f32(a_data[i]) * insight::f16_to_f32(b_data[i]);
      *dst = insight::f32_to_f16(sum);
    } else {
      for (int i = 0; i < n; ++i)
        sum +=
            insight::bf16_to_f32(a_data[i]) * insight::bf16_to_f32(b_data[i]);
      *dst = insight::f32_to_bf16(sum);
    }
  } else {
    double *c = (double *)out->data;
#if HAVE_BLAS_ACCEL
    *c = cblas_ddot(n, (double *)a->data, 1, (double *)b->data, 1);
#else
    *c = dot_f64_fallback((double *)a->data, (double *)b->data, n);
#endif
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(dot, INSIGHT_DTYPE_F32, dot_kernel_cpu);
REGISTER_CPU_KERNEL(dot, INSIGHT_DTYPE_F64, dot_kernel_cpu);
REGISTER_CPU_KERNEL(dot, INSIGHT_DTYPE_F16, dot_kernel_cpu);
REGISTER_CPU_KERNEL(dot, INSIGHT_DTYPE_BF16, dot_kernel_cpu);
