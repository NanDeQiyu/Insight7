// backends/cpu/kernels/signal/bsplines/gauss_spline.cpp
// CPU kernel for Gaussian approximation to B-spline
// gauss_spline(x, n) = 1/sqrt(2*pi*sigma^2) * exp(-x^2 / (2*sigma^2))
// where sigma^2 = (n+1)/12
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

// gauss_spline kernel
// inputs: [0]=r (1-element F64 or I32, spline order n)
// outputs: [0]=result (F64), pre-filled with x values, overwritten in-place
C_Status gauss_spline_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *r_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!r_arr || !out) {
    cpu_set_last_error("gauss_spline: null array pointer");
    return C_FAILED;
  }

  if (!r_arr->data || r_arr->numel < 1) {
    cpu_set_last_error("gauss_spline: order array is empty");
    return C_FAILED;
  }

  // Read order parameter (could be F64 or I32)
  double n;
  if (r_arr->dtype == INSIGHT_DTYPE_F64) {
    n = *(const double *)r_arr->data;
  } else if (r_arr->dtype == INSIGHT_DTYPE_I32) {
    n = (double)(*(const int32_t *)r_arr->data);
  } else if (r_arr->dtype == INSIGHT_DTYPE_I64) {
    n = (double)(*(const int64_t *)r_arr->data);
  } else if (r_arr->dtype == INSIGHT_DTYPE_F32) {
    n = (double)(*(const float *)r_arr->data);
  } else {
    cpu_set_last_error("gauss_spline: unsupported order dtype");
    return C_FAILED;
  }

  double sigma_sq = (n + 1.0) / 12.0;
  double r_sigma_sq = 0.5 / sigma_sq;
  double coeff = 1.0 / std::sqrt(2.0 * M_PI * sigma_sq);

  int64_t M = out->numel;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      double x2 = data[i] * data[i];
      data[i] = coeff * std::exp(-x2 * r_sigma_sq);
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    float coeff_f = (float)coeff;
    float r_sigma_sq_f = (float)r_sigma_sq;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      float x2 = data[i] * data[i];
      data[i] = coeff_f * std::exp(-x2 * r_sigma_sq_f);
    }
  } else {
    cpu_set_last_error("gauss_spline: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(gauss_spline, INSIGHT_DTYPE_F64, gauss_spline_kernel_cpu);
REGISTER_CPU_KERNEL(gauss_spline, INSIGHT_DTYPE_F32, gauss_spline_kernel_cpu);
