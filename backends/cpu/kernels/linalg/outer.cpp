// backends/cpu/kernels/linalg/outer.cpp
/**
 * @file outer.cpp
 * @brief CPU kernel for outer product of two 1D vectors.
 */

#include "common.h"

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
