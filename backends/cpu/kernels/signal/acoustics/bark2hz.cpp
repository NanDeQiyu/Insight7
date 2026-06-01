// backends/cpu/kernels/signal/acoustics/bark2hz.cpp
// CPU kernel for Bark to Hz conversion.
// Uses Newton's method to invert: bark = 13*atan(0.00076*hz)
// + 3.5*atan((hz/7500)^2) Initial guess: hz = 600 * sinh(bark / 3.0)
// (Traunmuller approximation) Then refine with 5 Newton iterations. inputs[0]:
// bark array (F64) outputs[0]: hz array (F64)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward bark function: hz -> bark
static inline double bark_func(double hz) {
  double ratio = hz / 7500.0;
  return 13.0 * std::atan(0.00076 * hz) + 3.5 * std::atan(ratio * ratio);
}

// Derivative: d(bark)/d(hz)
static inline double bark_deriv(double hz) {
  double t1 = 0.00076 * hz;
  double t2 = hz / 7500.0;
  return 13.0 * 0.00076 / (1.0 + t1 * t1) +
         3.5 * 2.0 * t2 * (1.0 / 7500.0) / (1.0 + t2 * t2 * t2 * t2);
}

extern "C" {

C_Status signal_bark2hz_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *bark_arr = (InsightArray *)inputs[0];
  InsightArray *hz_arr = (InsightArray *)outputs[0];

  if (!bark_arr || !hz_arr) {
    cpu_set_last_error("signal_bark2hz: null array pointer");
    return C_FAILED;
  }

  if (!bark_arr->data || !hz_arr->data) {
    cpu_set_last_error("signal_bark2hz: null data pointer");
    return C_FAILED;
  }

  int64_t n = bark_arr->numel;

  switch (bark_arr->dtype) {
  case INSIGHT_DTYPE_F64: {
    const double *bark = (const double *)bark_arr->data;
    double *hz = (double *)hz_arr->data;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        double b = bark[i];
        // Traunmuller initial guess
        double h = 600.0 * std::sinh(b / 3.0);
        // Newton's method: 5 iterations
        for (int iter = 0; iter < 5; ++iter) {
          double f = bark_func(h) - b;
          double df = bark_deriv(h);
          if (std::abs(df) < 1e-30)
            break;
          h -= f / df;
          if (h < 0.0)
            h = 0.0;
        }
        hz[i] = h;
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        double b = bark[i];
        double h = 600.0 * std::sinh(b / 3.0);
        for (int iter = 0; iter < 5; ++iter) {
          double f = bark_func(h) - b;
          double df = bark_deriv(h);
          if (std::abs(df) < 1e-30)
            break;
          h -= f / df;
          if (h < 0.0)
            h = 0.0;
        }
        hz[i] = h;
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_F32: {
    const float *bark = (const float *)bark_arr->data;
    float *hz = (float *)hz_arr->data;

#ifdef _OPENMP
    if (n > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < n; ++i) {
        float b = bark[i];
        float h = 600.0f * std::sinh(b / 3.0f);
        for (int iter = 0; iter < 5; ++iter) {
          double hd = (double)h;
          double f = bark_func(hd) - (double)b;
          double df = bark_deriv(hd);
          if (std::abs(df) < 1e-30)
            break;
          h -= (float)(f / df);
          if (h < 0.0f)
            h = 0.0f;
        }
        hz[i] = h;
      }
    } else {
#endif
      for (int64_t i = 0; i < n; ++i) {
        float b = bark[i];
        float h = 600.0f * std::sinh(b / 3.0f);
        for (int iter = 0; iter < 5; ++iter) {
          double hd = (double)h;
          double f = bark_func(hd) - (double)b;
          double df = bark_deriv(hd);
          if (std::abs(df) < 1e-30)
            break;
          h -= (float)(f / df);
          if (h < 0.0f)
            h = 0.0f;
        }
        hz[i] = h;
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  default:
    cpu_set_last_error("signal_bark2hz: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_bark2hz, INSIGHT_DTYPE_F64,
                    signal_bark2hz_kernel_cpu);
REGISTER_CPU_KERNEL(signal_bark2hz, INSIGHT_DTYPE_F32,
                    signal_bark2hz_kernel_cpu);
