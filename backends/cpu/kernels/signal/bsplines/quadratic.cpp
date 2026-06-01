// backends/cpu/kernels/signal/bsplines/quadratic.cpp
// CPU kernel for quadratic B-spline basis function
// |x| < 0.5:     3/4 - x^2
// 0.5 <= |x| < 1.5: (3-2*|x|)^2 / 8
// |x| >= 1.5:    0
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

// quadratic bspline kernel
// Same pattern as cubic: output buffer pre-filled with x values,
// overwritten in-place with B-spline values.
C_Status quadratic_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("quadratic: null array pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      double ax = std::fabs(data[i]);
      if (ax < 0.5) {
        data[i] = 0.75 - ax * ax;
      } else if (ax < 1.5) {
        double t = 3.0 - 2.0 * ax;
        data[i] = (t * t) / 8.0;
      } else {
        data[i] = 0.0;
      }
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      float ax = std::fabs(data[i]);
      if (ax < 0.5f) {
        data[i] = 0.75f - ax * ax;
      } else if (ax < 1.5f) {
        float t = 3.0f - 2.0f * ax;
        data[i] = (t * t) / 8.0f;
      } else {
        data[i] = 0.0f;
      }
    }
  } else {
    cpu_set_last_error("quadratic: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(quadratic, INSIGHT_DTYPE_F64, quadratic_kernel_cpu);
REGISTER_CPU_KERNEL(quadratic, INSIGHT_DTYPE_F32, quadratic_kernel_cpu);
