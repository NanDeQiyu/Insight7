// backends/cpu/kernels/signal/windows/kaiser.cpp
// CPU kernel for Kaiser window generation.
// w[n] = I0(beta * sqrt(1 - ((n - (M-1)/2) / ((M-1)/2))^2)) / I0(beta)
// where I0 is the modified Bessel function of the first kind, order 0.
// inputs[0]: beta (1-element F64 array)
// outputs[0]: window output
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// Modified Bessel function I0(x) using Abramowitz & Stegun approximation.
// Accurate to ~1.6e-7 for all x >= 0, sufficient for window generation.
static inline double bessel_i0(double x) {
  double ax = fabs(x);
  if (ax <= 3.75) {
    double t = x / 3.75;
    double t2 = t * t;
    return 1.0 + 3.5156229 * t2 + 3.0899424 * t2 * t2 +
           1.2067492 * t2 * t2 * t2 + 0.2659732 * t2 * t2 * t2 * t2 +
           0.0360768 * t2 * t2 * t2 * t2 * t2 +
           0.0045813 * t2 * t2 * t2 * t2 * t2 * t2;
  } else {
    double t = 3.75 / ax;
    return (exp(ax) / sqrt(ax)) *
           (0.39894228 + 0.01328592 * t + 0.000225319 * t * t -
            0.00157565 * t * t * t + 0.00916281 * t * t * t * t -
            0.02057706 * t * t * t * t * t +
            0.02635537 * t * t * t * t * t * t -
            0.01647633 * t * t * t * t * t * t * t +
            0.00392377 * t * t * t * t * t * t * t * t);
  }
}

extern "C" {

C_Status kaiser_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *beta_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!beta_arr || !out) {
    cpu_set_last_error("kaiser: null array pointer");
    return C_FAILED;
  }

  double beta = *(double *)beta_arr->data;
  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    double i0_beta = bessel_i0(beta);
    if (i0_beta == 0.0) {
      for (int64_t i = 0; i < M; ++i)
        data[i] = 0.0;
      return C_SUCCESS;
    }
    double half = (M - 1) / 2.0;
    double inv_half = (half > 0.0) ? 1.0 / half : 0.0;
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
    for (int64_t i = 0; i < M; ++i) {
      double ratio = (i - half) * inv_half;
      double arg = beta * sqrt(fmax(0.0, 1.0 - ratio * ratio));
      data[i] = bessel_i0(arg) / i0_beta;
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    float beta_f = (float)beta;
    // Reuse double-precision bessel_i0 for accuracy, cast results to float
    double i0_beta = bessel_i0(beta);
    if (i0_beta == 0.0) {
      for (int64_t i = 0; i < M; ++i)
        data[i] = 0.0f;
      return C_SUCCESS;
    }
    float half = (M - 1) / 2.0f;
    float inv_half = (half > 0.0f) ? 1.0f / half : 0.0f;
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
    for (int64_t i = 0; i < M; ++i) {
      float ratio = (i - half) * inv_half;
      float arg = beta_f * sqrtf(fmaxf(0.0f, 1.0f - ratio * ratio));
      data[i] = (float)(bessel_i0((double)arg) / i0_beta);
    }
  } else {
    cpu_set_last_error("kaiser: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(kaiser, INSIGHT_DTYPE_F64, kaiser_kernel_cpu);
REGISTER_CPU_KERNEL(kaiser, INSIGHT_DTYPE_F32, kaiser_kernel_cpu);
