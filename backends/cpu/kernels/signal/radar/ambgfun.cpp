// backends/cpu/kernels/signal/radar/ambgfun.cpp
// CPU kernel for Ambiguity Function computation
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

// ambgfun kernel
// inputs: [0]=x_data (complex128, normalized), [1]=y_data (complex128,
// normalized),
//         [2]=params (float64 array: [fs, N, M])
// outputs: [0]=ambg (float64, doppler_len x delay_len)
//
// The frontend converts input to complex128, normalizes, and passes parameters.
C_Status ambgfun_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *y_arr = (InsightArray *)inputs[1];
  InsightArray *params_arr = (InsightArray *)inputs[2];
  InsightArray *out_arr = (InsightArray *)outputs[0];

  if (!x_arr || !y_arr || !params_arr || !out_arr) {
    cpu_set_last_error("ambgfun: null array pointer");
    return C_FAILED;
  }

  const std::complex<double> *x_data =
      (const std::complex<double> *)x_arr->data;
  const std::complex<double> *y_data =
      (const std::complex<double> *)y_arr->data;
  const double *params = (const double *)params_arr->data;
  double *ambg = (double *)out_arr->data;

  double fs = params[0];
  int64_t N = (int64_t)params[1];
  int64_t M = (int64_t)params[2];

  int64_t delay_len = N + M - 1;
  int64_t doppler_len = N;

  if (!x_data || !y_data || !ambg) {
    cpu_set_last_error("ambgfun: null data pointer");
    return C_FAILED;
  }

  // Main computation: triple nested loop
  // Parallelize over doppler (outer loop) if large enough
#pragma omp parallel for if (doppler_len > 32)
  for (int64_t fd = 0; fd < doppler_len; ++fd) {
    double freq = (fd - doppler_len / 2.0) * fs / doppler_len;
    double omega = 2.0 * M_PI * freq / fs;

    for (int64_t tau = 0; tau < delay_len; ++tau) {
      int64_t tau_shift = tau - (M - 1);

      std::complex<double> sum_val(0.0, 0.0);
      for (int64_t n = 0; n < N; ++n) {
        int64_t m = n - tau_shift;
        if (m >= 0 && m < M) {
          double phase = omega * n;
          sum_val += x_data[n] * std::conj(y_data[m]) *
                     std::exp(std::complex<double>(0.0, phase));
        }
      }
      ambg[fd * delay_len + tau] = std::abs(sum_val);
    }
  }

  // Normalize peak to 1
  double peak = 0.0;
  int64_t total = doppler_len * delay_len;
  for (int64_t i = 0; i < total; ++i) {
    if (ambg[i] > peak)
      peak = ambg[i];
  }
  if (peak > 0) {
    for (int64_t i = 0; i < total; ++i) {
      ambg[i] /= peak;
    }
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(ambgfun, INSIGHT_DTYPE_F64, ambgfun_kernel_cpu);
