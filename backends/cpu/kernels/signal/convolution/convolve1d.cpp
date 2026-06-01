// backends/cpu/kernels/signal/convolution/convolve1d.cpp
// CPU kernel for 1D direct convolution
// y[n] = sum_{k=0}^{nh-1} x[n+k-offset] * h[k]
// mode: 0=valid, 1=same, 2=full
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status convolve1d_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *h = (InsightArray *)inputs[1];
  int32_t mode = *(int32_t *)((InsightArray *)inputs[2])->data;
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !h || !out) {
    cpu_set_last_error("convolve1d: null array pointer");
    return C_FAILED;
  }

  int64_t nx = x->numel;
  int64_t nh = h->numel;
  int64_t out_len, offset;

  if (mode == 0) { // valid
    out_len = nx - nh + 1;
    offset = 0;
  } else if (mode == 1) { // same
    out_len = nx;
    offset = nh / 2;
  } else { // full
    out_len = nx + nh - 1;
    offset = 0;
  }

  if (x->dtype == INSIGHT_DTYPE_F64) {
    const double *xd = (const double *)x->data;
    const double *hd = (const double *)h->data;
    double *od = (double *)out->data;
#pragma omp parallel for if (out_len > 1000)
    for (int64_t i = 0; i < out_len; ++i) {
      double sum = 0.0;
      for (int64_t j = 0; j < nh; ++j) {
        int64_t idx = i + j - offset;
        if (idx >= 0 && idx < nx) {
          sum += xd[idx] * hd[j];
        }
      }
      od[i] = sum;
    }
  } else if (x->dtype == INSIGHT_DTYPE_F32) {
    const float *xd = (const float *)x->data;
    const float *hd = (const float *)h->data;
    float *od = (float *)out->data;
#pragma omp parallel for if (out_len > 1000)
    for (int64_t i = 0; i < out_len; ++i) {
      float sum = 0.0f;
      for (int64_t j = 0; j < nh; ++j) {
        int64_t idx = i + j - offset;
        if (idx >= 0 && idx < nx) {
          sum += xd[idx] * hd[j];
        }
      }
      od[i] = sum;
    }
  } else {
    cpu_set_last_error("convolve1d: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(convolve1d, INSIGHT_DTYPE_F64, convolve1d_kernel_cpu);
REGISTER_CPU_KERNEL(convolve1d, INSIGHT_DTYPE_F32, convolve1d_kernel_cpu);
