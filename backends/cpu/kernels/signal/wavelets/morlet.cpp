// backends/cpu/kernels/signal/wavelets/morlet.cpp
// CPU kernel for complex Morlet wavelet.
// w[n] = exp(j*2*pi*w0*t) * exp(-t^2/2) where t = n - (M-1)/2
// inputs[0]: w0 (center frequency, F64, 1-element)
// inputs[1]: s (scale, F64, 1-element)
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

C_Status signal_morlet_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *w0_arr = (InsightArray *)inputs[0];
  InsightArray *s_arr = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!w0_arr || !s_arr || !out) {
    cpu_set_last_error("signal_morlet: null array pointer");
    return C_FAILED;
  }

  if (!w0_arr->data || !s_arr->data || !out->data) {
    cpu_set_last_error("signal_morlet: null data pointer");
    return C_FAILED;
  }

  double w0 = *(const double *)w0_arr->data;
  double s = *(const double *)s_arr->data;
  int64_t M = out->numel;

  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_C64) {
    std::complex<double> *data =
        reinterpret_cast<std::complex<double> *>(out->data);
    double half = (double)(M - 1) / 2.0;

#ifdef _OPENMP
    if (M > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < M; ++i) {
        double t = ((double)i - half) / s;
        double gauss = std::exp(-0.5 * t * t);
        double phase = 2.0 * M_PI * w0 * t;
        data[i] = std::complex<double>(gauss * std::cos(phase),
                                       gauss * std::sin(phase));
      }
    } else {
#endif
      for (int64_t i = 0; i < M; ++i) {
        double t = ((double)i - half) / s;
        double gauss = std::exp(-0.5 * t * t);
        double phase = 2.0 * M_PI * w0 * t;
        data[i] = std::complex<double>(gauss * std::cos(phase),
                                       gauss * std::sin(phase));
      }
#ifdef _OPENMP
    }
#endif
  } else {
    cpu_set_last_error("signal_morlet: only C64 dtype supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_morlet, INSIGHT_DTYPE_C64, signal_morlet_kernel_cpu);
