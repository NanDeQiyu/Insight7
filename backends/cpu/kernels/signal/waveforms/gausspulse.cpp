// backends/cpu/kernels/signal/waveforms/gausspulse.cpp
// CPU kernel for Gaussian-modulated sinusoidal pulse
// env = exp(-t^2 / (2*fc^2)), I = env * cos(2*pi*fc*t)
// Uses bandwidth-derived envelope: a = (pi*fc*bw)^2 / (4*log10(10^(bwr/20)))
// Simplified: env = exp(a*t^2), output = env * cos(2*pi*fc*t)
// Here a = -4*log(2)*pi^2*fc^2*bw^2 (from bwr=-6dB reference)
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

// gausspulse kernel
// inputs: [0]=fc (center freq, 1-elem F64), [1]=bw (bandwidth, 1-elem F64)
// outputs: [0]=in-phase component (F64)
C_Status gausspulse_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *fc_arr = (InsightArray *)inputs[0];
  InsightArray *bw_arr = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!fc_arr || !bw_arr || !out) {
    cpu_set_last_error("gausspulse: null array pointer");
    return C_FAILED;
  }

  if (!fc_arr->data || !bw_arr->data) {
    cpu_set_last_error("gausspulse: parameter array data is null");
    return C_FAILED;
  }

  double fc = *(const double *)fc_arr->data;
  double bw = *(const double *)bw_arr->data;
  int64_t M = out->numel;

  // Envelope parameter: a = -4*ln(2)*pi^2*fc^2*bw^2
  // (derived from -6dB bandwidth reference: bwr=-6)
  double a = -4.0 * std::log(2.0) * M_PI * M_PI * fc * fc * bw * bw;
  double two_pi_fc = 2.0 * M_PI * fc;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    double inv_M1 = (M > 1) ? 1.0 / (double)(M - 1) : 1.0;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      double t = (double)i * inv_M1;
      double t2 = t * t;
      double env = std::exp(a * t2);
      data[i] = env * std::cos(two_pi_fc * t);
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    float inv_M1 = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
    float af = (float)a;
    float two_pi_fc_f = (float)two_pi_fc;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      float t = (float)i * inv_M1;
      float t2 = t * t;
      float env = std::exp(af * t2);
      data[i] = env * std::cos(two_pi_fc_f * t);
    }
  } else {
    cpu_set_last_error("gausspulse: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(gausspulse, INSIGHT_DTYPE_F64, gausspulse_kernel_cpu);
REGISTER_CPU_KERNEL(gausspulse, INSIGHT_DTYPE_F32, gausspulse_kernel_cpu);
