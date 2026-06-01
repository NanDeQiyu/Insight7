// backends/cpu/kernels/signal/peak_finding/boolrelextrema_1d.cpp
// CPU kernel for 1D relative extrema detection
// For each element, checks if it is a local max/min by comparing with
// 'order' neighbors on each side.
// inputs: [0]=data (1D), [1]=order (I32 scalar), [2]=comparator (I32:
// 0=greater, 1=less) outputs: [0]=mask (bool, same shape as data)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status boolrelextrema_1d_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *data = (InsightArray *)inputs[0];
  int32_t order = *(int32_t *)((InsightArray *)inputs[1])->data;
  int32_t cmp = *(int32_t *)((InsightArray *)inputs[2])->data;
  InsightArray *out = (InsightArray *)outputs[0];

  if (!data || !out) {
    cpu_set_last_error("boolrelextrema_1d: null array pointer");
    return C_FAILED;
  }

  int64_t n = data->numel;

  if (data->dtype == INSIGHT_DTYPE_F64) {
    const double *dd = (const double *)data->data;
    bool *od = (bool *)out->data;
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; ++i) {
      bool is_extremum = true;
      for (int64_t j = 1; j <= order; ++j) {
        if (i - j >= 0) {
          if (cmp == 0 && dd[i] < dd[i - j]) {
            is_extremum = false;
            break;
          }
          if (cmp == 1 && dd[i] > dd[i - j]) {
            is_extremum = false;
            break;
          }
        }
        if (i + j < n) {
          if (cmp == 0 && dd[i] < dd[i + j]) {
            is_extremum = false;
            break;
          }
          if (cmp == 1 && dd[i] > dd[i + j]) {
            is_extremum = false;
            break;
          }
        }
      }
      od[i] = is_extremum;
    }
  } else if (data->dtype == INSIGHT_DTYPE_F32) {
    const float *dd = (const float *)data->data;
    bool *od = (bool *)out->data;
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; ++i) {
      bool is_extremum = true;
      for (int64_t j = 1; j <= order; ++j) {
        if (i - j >= 0) {
          if (cmp == 0 && dd[i] < dd[i - j]) {
            is_extremum = false;
            break;
          }
          if (cmp == 1 && dd[i] > dd[i - j]) {
            is_extremum = false;
            break;
          }
        }
        if (i + j < n) {
          if (cmp == 0 && dd[i] < dd[i + j]) {
            is_extremum = false;
            break;
          }
          if (cmp == 1 && dd[i] > dd[i + j]) {
            is_extremum = false;
            break;
          }
        }
      }
      od[i] = is_extremum;
    }
  } else {
    cpu_set_last_error("boolrelextrema_1d: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(boolrelextrema_1d, INSIGHT_DTYPE_F64,
                    boolrelextrema_1d_kernel_cpu);
REGISTER_CPU_KERNEL(boolrelextrema_1d, INSIGHT_DTYPE_F32,
                    boolrelextrema_1d_kernel_cpu);
