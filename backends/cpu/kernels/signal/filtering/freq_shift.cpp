// backends/cpu/kernels/signal/filtering/freq_shift.cpp
// CPU kernel for frequency shift.
// y[n] = x[n] * exp(j * 2 * pi * freq * n / fs)
// For real input, this produces a complex analytic signal.
// inputs[0]: data array (F64 or F32)
// inputs[1]: freq (F64, 1-element)
// inputs[2]: fs (F64, 1-element)
// outputs[0]: shifted array (same real dtype as input)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <complex>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status signal_freq_shift_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *freq_arr = (InsightArray *)inputs[1];
  InsightArray *fs_arr = (InsightArray *)inputs[2];
  InsightArray *y_arr = (InsightArray *)outputs[0];

  if (!x_arr || !freq_arr || !fs_arr || !y_arr) {
    cpu_set_last_error("signal_freq_shift: null array pointer");
    return C_FAILED;
  }

  if (!x_arr->data || !freq_arr->data || !fs_arr->data || !y_arr->data) {
    cpu_set_last_error("signal_freq_shift: null data pointer");
    return C_FAILED;
  }

  double freq = *(const double *)freq_arr->data;
  double fs = *(const double *)fs_arr->data;
  int64_t n = x_arr->numel;

  if (fs == 0.0) {
    cpu_set_last_error("signal_freq_shift: fs must be non-zero");
    return C_FAILED;
  }

  double phase_inc = 2.0 * M_PI * freq / fs;

  switch (x_arr->dtype) {
  case INSIGHT_DTYPE_F64: {
    const double *x = (const double *)x_arr->data;
    double *y = (double *)y_arr->data;

    // Real output: y[n] = x[n] * cos(2*pi*freq*n/fs)
    // (Real part of the complex modulation)
#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        double phase = phase_inc * (double)i;
        y[i] = x[i] * std::cos(phase);
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        double phase = phase_inc * (double)i;
        y[i] = x[i] * std::cos(phase);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_F32: {
    const float *x = (const float *)x_arr->data;
    float *y = (float *)y_arr->data;
    float phase_inc_f = (float)phase_inc;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        float phase = phase_inc_f * (float)i;
        y[i] = x[i] * std::cos(phase);
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        float phase = phase_inc_f * (float)i;
        y[i] = x[i] * std::cos(phase);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_C64: {
    const std::complex<double> *x =
        reinterpret_cast<const std::complex<double> *>(x_arr->data);
    std::complex<double> *y =
        reinterpret_cast<std::complex<double> *>(y_arr->data);

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        double phase = phase_inc * (double)i;
        y[i] = x[i] * std::exp(std::complex<double>(0.0, phase));
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        double phase = phase_inc * (double)i;
        y[i] = x[i] * std::exp(std::complex<double>(0.0, phase));
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_C32: {
    const std::complex<float> *x =
        reinterpret_cast<const std::complex<float> *>(x_arr->data);
    std::complex<float> *y =
        reinterpret_cast<std::complex<float> *>(y_arr->data);
    float phase_inc_f = (float)phase_inc;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        float phase = phase_inc_f * (float)i;
        y[i] = x[i] * std::exp(std::complex<float>(0.0f, phase));
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        float phase = phase_inc_f * (float)i;
        y[i] = x[i] * std::exp(std::complex<float>(0.0f, phase));
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  default:
    cpu_set_last_error(
        "signal_freq_shift: unsupported dtype, need F32, F64, C32, or C64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_freq_shift, INSIGHT_DTYPE_F64,
                    signal_freq_shift_kernel_cpu);
REGISTER_CPU_KERNEL(signal_freq_shift, INSIGHT_DTYPE_F32,
                    signal_freq_shift_kernel_cpu);
REGISTER_CPU_KERNEL(signal_freq_shift, INSIGHT_DTYPE_C64,
                    signal_freq_shift_kernel_cpu);
REGISTER_CPU_KERNEL(signal_freq_shift, INSIGHT_DTYPE_C32,
                    signal_freq_shift_kernel_cpu);
