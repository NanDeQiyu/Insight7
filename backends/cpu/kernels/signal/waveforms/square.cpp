// backends/cpu/kernels/signal/waveforms/square.cpp
// CPU kernel for square wave waveform
// w[n] = 1 if sin(2*pi*n/(M-1)) >= 0 else -1  (with duty parameter)
// Simplified: w[n] = 1 if (n mod period) < duty*period else -1
// Uses phase approach: w[n] = 1 if (n/(M-1)) mod 1.0 < duty else -1
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

// signal_square kernel (renamed to avoid collision with unary square)
// inputs: [0]=duty (1-element F64, default 0.5)
// outputs: [0]=waveform (F64)
C_Status signal_square_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *duty_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!duty_arr || !out) {
    cpu_set_last_error("square: null array pointer");
    return C_FAILED;
  }

  if (!duty_arr->data || duty_arr->numel < 1) {
    cpu_set_last_error("square: duty array is empty");
    return C_FAILED;
  }

  double duty = *(const double *)duty_arr->data;
  int64_t M = out->numel;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    double inv_M = (M > 1) ? 1.0 / (double)(M - 1) : 1.0;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      double phase = std::fmod((double)i * inv_M, 1.0);
      data[i] = (phase < duty) ? 1.0 : -1.0;
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    float inv_M = (M > 1) ? 1.0f / (float)(M - 1) : 1.0f;
    float duty_f = (float)duty;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      float phase = std::fmod((float)i * inv_M, 1.0f);
      data[i] = (phase < duty_f) ? 1.0f : -1.0f;
    }
  } else {
    cpu_set_last_error("square: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_square, INSIGHT_DTYPE_F64, signal_square_kernel_cpu);
REGISTER_CPU_KERNEL(signal_square, INSIGHT_DTYPE_F32, signal_square_kernel_cpu);
