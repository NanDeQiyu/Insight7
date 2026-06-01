// backends/cpu/kernels/signal/bsplines/cubic.cpp
// CPU kernel for cubic B-spline basis function
// |x| < 1:       2/3 - |x|^2*(2-|x|)/2
// 1 <= |x| < 2:  (2-|x|)^3 / 6
// |x| >= 2:      0
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

// cubic bspline kernel
// inputs: none (parameters embedded in formula)
// outputs: [0]=result (F64), reads x from output array's pre-filled input
// NOTE: This kernel takes x-values as the pre-allocated output buffer
// and overwrites it with the B-spline values. The frontend will copy
// the input x to the output before calling this kernel.
C_Status cubic_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("cubic: null array pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      double ax = std::fabs(data[i]);
      if (ax < 1.0) {
        double ax2 = ax * ax;
        data[i] = (2.0 / 3.0) - 0.5 * ax2 * (2.0 - ax);
      } else if (ax < 2.0) {
        double t = 2.0 - ax;
        data[i] = (t * t * t) / 6.0;
      } else {
        data[i] = 0.0;
      }
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      float ax = std::fabs(data[i]);
      if (ax < 1.0f) {
        float ax2 = ax * ax;
        data[i] = (2.0f / 3.0f) - 0.5f * ax2 * (2.0f - ax);
      } else if (ax < 2.0f) {
        float t = 2.0f - ax;
        data[i] = (t * t * t) / 6.0f;
      } else {
        data[i] = 0.0f;
      }
    }
  } else {
    cpu_set_last_error("cubic: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(cubic, INSIGHT_DTYPE_F64, cubic_kernel_cpu);
REGISTER_CPU_KERNEL(cubic, INSIGHT_DTYPE_F32, cubic_kernel_cpu);
