// backends/cpu/kernels/linalg/outer.cpp
/**
 * @file outer.cpp
 * @brief CPU kernel for outer product of two 1D vectors.
 */

#include "../common/half_utils.h"
#include "common.h"

#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

static void outer_f32(const float *a, const float *b, float *c, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      c[i * n + j] = a[i] * b[j];
    }
  }
}

static void outer_f64(const double *a, const double *b, double *c, int m,
                      int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      c[i * n + j] = a[i] * b[j];
    }
  }
}

C_Status outer_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *a = (InsightArray *)inputs[1];
  InsightArray *b = (InsightArray *)inputs[2];

  if (!out || !a || !b) {
    cpu_set_last_error("outer: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(a) || !cpu_ensure_contiguous(b) ||
      !cpu_ensure_contiguous(out))
    return C_FAILED;

  int m = (int)a->numel;
  int n = (int)b->numel;

  if (out->dtype == INSIGHT_DTYPE_F32) {
    outer_f32((float *)a->data, (float *)b->data, (float *)out->data, m, n);
  } else if (out->dtype == INSIGHT_DTYPE_F16 ||
             out->dtype == INSIGHT_DTYPE_BF16) {
    const uint16_t *a_data = (const uint16_t *)a->data;
    const uint16_t *b_data = (const uint16_t *)b->data;
    uint16_t *dst = (uint16_t *)out->data;
    if (out->dtype == INSIGHT_DTYPE_F16) {
      for (int i = 0; i < m; ++i) {
        float ai = insight::f16_to_f32(a_data[i]);
        for (int j = 0; j < n; ++j) {
          dst[i * n + j] =
              insight::f32_to_f16(ai * insight::f16_to_f32(b_data[j]));
        }
      }
    } else {
      for (int i = 0; i < m; ++i) {
        float ai = insight::bf16_to_f32(a_data[i]);
        for (int j = 0; j < n; ++j) {
          dst[i * n + j] =
              insight::f32_to_bf16(ai * insight::bf16_to_f32(b_data[j]));
        }
      }
    }
  } else {
    outer_f64((double *)a->data, (double *)b->data, (double *)out->data, m, n);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(outer, INSIGHT_DTYPE_F32, outer_kernel_cpu);
REGISTER_CPU_KERNEL(outer, INSIGHT_DTYPE_F64, outer_kernel_cpu);
REGISTER_CPU_KERNEL(outer, INSIGHT_DTYPE_F16, outer_kernel_cpu);
REGISTER_CPU_KERNEL(outer, INSIGHT_DTYPE_BF16, outer_kernel_cpu);
