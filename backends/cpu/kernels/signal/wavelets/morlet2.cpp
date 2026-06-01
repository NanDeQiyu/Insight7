// backends/cpu/kernels/signal/wavelets/morlet2.cpp
// CPU kernel for Morlet wavelet (CWT-compatible).
// w[n] = exp(j*2*pi*s*t) * exp(-t^2/2) / sqrt(2*pi)
// where t = n - (M-1)/2, s = center frequency, w = width
// inputs[0]: s (center frequency, F64, 1-element)
// inputs[1]: w (width, F64, 1-element)
// outputs[0]: complex wavelet (C64)
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

C_Status signal_morlet2_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *s_arr = (InsightArray *)inputs[0];
  InsightArray *w_arr = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!s_arr || !w_arr || !out) {
    cpu_set_last_error("signal_morlet2: null array pointer");
    return C_FAILED;
  }

  if (!s_arr->data || !w_arr->data || !out->data) {
    cpu_set_last_error("signal_morlet2: null data pointer");
    return C_FAILED;
  }

  double s_val = *(const double *)s_arr->data;
  double w_val = *(const double *)w_arr->data;
  int64_t M = out->numel;

  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_C64) {
    std::complex<double> *data =
        reinterpret_cast<std::complex<double> *>(out->data);
    double half = (double)(M - 1) / 2.0;
    double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);

#ifdef _OPENMP
    if (M > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < M; ++i) {
        double t = (double)i - half;
        double gauss = std::exp(-0.5 * t * t) * inv_sqrt2pi;
        double phase = 2.0 * M_PI * s_val * t;
        data[i] = std::complex<double>(gauss * std::cos(phase),
                                       gauss * std::sin(phase));
      }
    } else {
#endif
      for (int64_t i = 0; i < M; ++i) {
        double t = (double)i - half;
        double gauss = std::exp(-0.5 * t * t) * inv_sqrt2pi;
        double phase = 2.0 * M_PI * s_val * t;
        data[i] = std::complex<double>(gauss * std::cos(phase),
                                       gauss * std::sin(phase));
      }
#ifdef _OPENMP
    }
#endif
  } else {
    cpu_set_last_error("signal_morlet2: only C64 dtype supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_morlet2, INSIGHT_DTYPE_C64,
                    signal_morlet2_kernel_cpu);
