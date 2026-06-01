// backends/cpu/kernels/signal/waveforms/chirp.cpp
// CPU kernel for linear chirp signal
// w[n] = cos(2*pi * (f0 + (f1-f0)*t/2) * t) where t = n/(M-1) * T
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

// chirp kernel
// inputs: [0]=f0 (1-elem F64), [1]=f1 (1-elem F64), [2]=T (1-elem F64)
// outputs: [0]=chirp signal (F64)
C_Status chirp_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *f0_arr = (InsightArray *)inputs[0];
  InsightArray *f1_arr = (InsightArray *)inputs[1];
  InsightArray *T_arr = (InsightArray *)inputs[2];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!f0_arr || !f1_arr || !T_arr || !out) {
    cpu_set_last_error("chirp: null array pointer");
    return C_FAILED;
  }

  if (!f0_arr->data || !f1_arr->data || !T_arr->data) {
    cpu_set_last_error("chirp: parameter array data is null");
    return C_FAILED;
  }

  double f0 = *(const double *)f0_arr->data;
  double f1 = *(const double *)f1_arr->data;
  double T = *(const double *)T_arr->data;
  int64_t M = out->numel;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    double inv_M1 = (M > 1) ? 1.0 / (double)(M - 1) : 1.0;
    double beta = (f1 - f0) / T;
    double two_pi = 2.0 * M_PI;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      double t = (double)i * inv_M1 * T;
      double inst_freq = f0 + beta * t;
      data[i] = std::cos(two_pi * inst_freq * t);
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    float inv_M1 = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
    float f0f = (float)f0, f1f = (float)f1, Tf = (float)T;
    float beta = (f1f - f0f) / Tf;
    float two_pi = 2.0f * (float)M_PI;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      float t = (float)i * inv_M1 * Tf;
      float inst_freq = f0f + beta * t;
      data[i] = std::cos(two_pi * inst_freq * t);
    }
  } else {
    cpu_set_last_error("chirp: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(chirp, INSIGHT_DTYPE_F64, chirp_kernel_cpu);
REGISTER_CPU_KERNEL(chirp, INSIGHT_DTYPE_F32, chirp_kernel_cpu);
