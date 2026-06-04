// backends/cpu/kernels/signal/filter_design/firwin.cpp
// CPU kernel for FIR filter windowed sinc design.
// h[n] = sinc(2*fc*(n - (M-1)/2)) * window[n] where fc is cutoff frequency
// The window function is a Hann window applied element-wise.
// inputs[0]: fc (cutoff frequency, F64, 1-element)
// inputs[1]: M (filter length, I64, 1-element)
// outputs[0]: FIR coefficients (F64)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1
static inline double sinc_func(double x) {
  if (std::abs(x) < 1e-15)
    return 1.0;
  return std::sin(M_PI * x) / (M_PI * x);
}

extern "C" {

C_Status signal_firwin_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *fc_arr = (InsightArray *)inputs[0];
  InsightArray *M_arr = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!fc_arr || !M_arr || !out) {
    cpu_set_last_error("signal_firwin: null array pointer");
    return C_FAILED;
  }

  if (!fc_arr->data || !M_arr->data || !out->data) {
    cpu_set_last_error("signal_firwin: null data pointer");
    return C_FAILED;
  }

  double fc = *(const double *)fc_arr->data;
  int64_t M_len = *(const int64_t *)M_arr->data;

  if (M_len <= 0) {
    cpu_set_last_error("signal_firwin: M must be positive");
    return C_FAILED;
  }

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    double half = (double)(M_len - 1) / 2.0;

#ifdef _OPENMP
    if (M_len > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < M_len; ++i) {
        double n_centered = (double)i - half;
        // sinc kernel
        double h = sinc_func(2.0 * fc * n_centered);
        // Hann window: 0.5 * (1 - cos(2*pi*n/(M-1)))
        double win =
            0.5 *
            (1.0 - std::cos(2.0 * M_PI * (double)i / (double)(M_len - 1)));
        data[i] = h * win;
      }
    } else {
#endif
      for (int64_t i = 0; i < M_len; ++i) {
        double n_centered = (double)i - half;
        double h = sinc_func(2.0 * fc * n_centered);
        double win =
            0.5 *
            (1.0 - std::cos(2.0 * M_PI * (double)i / (double)(M_len - 1)));
        data[i] = h * win;
      }
#ifdef _OPENMP
    }
#endif
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    float half = (float)(M_len - 1) / 2.0f;
    float fc_f = (float)fc;

#ifdef _OPENMP
    if (M_len > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < M_len; ++i) {
        float n_centered = (float)i - half;
        // sinc kernel (use float sinf)
        float arg = 2.0f * fc_f * n_centered;
        float h = (std::abs(arg) < 1e-15f)
                      ? 1.0f
                      : std::sin((float)M_PI * arg) / ((float)M_PI * arg);
        // Hann window
        float win = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i /
                                            (float)(M_len - 1)));
        data[i] = h * win;
      }
    } else {
#endif
      for (int64_t i = 0; i < M_len; ++i) {
        float n_centered = (float)i - half;
        float arg = 2.0f * fc_f * n_centered;
        float h = (std::abs(arg) < 1e-15f)
                      ? 1.0f
                      : std::sin((float)M_PI * arg) / ((float)M_PI * arg);
        float win = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i /
                                            (float)(M_len - 1)));
        data[i] = h * win;
      }
#ifdef _OPENMP
    }
#endif
  } else {
    cpu_set_last_error("signal_firwin: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_firwin, INSIGHT_DTYPE_F64, signal_firwin_kernel_cpu);
REGISTER_CPU_KERNEL(signal_firwin, INSIGHT_DTYPE_F32, signal_firwin_kernel_cpu);
