// backends/cpu/kernels/signal/acoustics/mel_frequencies.cpp
// CPU kernel for generating mel frequency points.
// Generate n_mels equally spaced points in mel scale between f_low and f_high.
// Steps: convert f_low/f_high to mel, linearly space, convert back to Hz.
// inputs[0]: n_mels (I64, 1-element)
// inputs[1]: f_low (F64, 1-element)
// inputs[2]: f_high (F64, 1-element)
// outputs[0]: frequency array (F64)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status signal_mel_frequencies_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *n_mels_arr = (InsightArray *)inputs[0];
  InsightArray *flo_arr = (InsightArray *)inputs[1];
  InsightArray *fhi_arr = (InsightArray *)inputs[2];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!n_mels_arr || !flo_arr || !fhi_arr || !out) {
    cpu_set_last_error("signal_mel_frequencies: null array pointer");
    return C_FAILED;
  }

  if (!n_mels_arr->data || !flo_arr->data || !fhi_arr->data || !out->data) {
    cpu_set_last_error("signal_mel_frequencies: null data pointer");
    return C_FAILED;
  }

  int64_t n_mels = *(const int64_t *)n_mels_arr->data;
  double f_low = *(const double *)flo_arr->data;
  double f_high = *(const double *)fhi_arr->data;

  if (n_mels <= 0) {
    cpu_set_last_error("signal_mel_frequencies: n_mels must be positive");
    return C_FAILED;
  }

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;

    // Convert endpoints to mel scale
    double mel_low = 2595.0 * std::log10(1.0 + f_low / 700.0);
    double mel_high = 2595.0 * std::log10(1.0 + f_high / 700.0);

    if (n_mels == 1) {
      // Single point: midpoint
      double mel_mid = (mel_low + mel_high) / 2.0;
      data[0] = 700.0 * (std::pow(10.0, mel_mid / 2595.0) - 1.0);
    } else {
      double mel_step = (mel_high - mel_low) / (double)(n_mels - 1);

#ifdef _OPENMP
      if (n_mels > 1000) {
#pragma omp parallel for
        for (int64_t i = 0; i < n_mels; ++i) {
          double mel_val = mel_low + (double)i * mel_step;
          data[i] = 700.0 * (std::pow(10.0, mel_val / 2595.0) - 1.0);
        }
      } else {
#endif
        for (int64_t i = 0; i < n_mels; ++i) {
          double mel_val = mel_low + (double)i * mel_step;
          data[i] = 700.0 * (std::pow(10.0, mel_val / 2595.0) - 1.0);
        }
#ifdef _OPENMP
      }
#endif
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;

    // Convert endpoints to mel scale
    float mel_low = 2595.0f * std::log10(1.0f + (float)f_low / 700.0f);
    float mel_high = 2595.0f * std::log10(1.0f + (float)f_high / 700.0f);

    if (n_mels == 1) {
      // Single point: midpoint
      float mel_mid = (mel_low + mel_high) / 2.0f;
      data[0] = 700.0f * (std::pow(10.0f, mel_mid / 2595.0f) - 1.0f);
    } else {
      float mel_step = (mel_high - mel_low) / (float)(n_mels - 1);

#ifdef _OPENMP
      if (n_mels > 1000) {
#pragma omp parallel for
        for (int64_t i = 0; i < n_mels; ++i) {
          float mel_val = mel_low + (float)i * mel_step;
          data[i] = 700.0f * (std::pow(10.0f, mel_val / 2595.0f) - 1.0f);
        }
      } else {
#endif
        for (int64_t i = 0; i < n_mels; ++i) {
          float mel_val = mel_low + (float)i * mel_step;
          data[i] = 700.0f * (std::pow(10.0f, mel_val / 2595.0f) - 1.0f);
        }
#ifdef _OPENMP
      }
#endif
    }
  } else {
    cpu_set_last_error(
        "signal_mel_frequencies: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_mel_frequencies, INSIGHT_DTYPE_F64,
                    signal_mel_frequencies_kernel_cpu);
REGISTER_CPU_KERNEL(signal_mel_frequencies, INSIGHT_DTYPE_F32,
                    signal_mel_frequencies_kernel_cpu);
