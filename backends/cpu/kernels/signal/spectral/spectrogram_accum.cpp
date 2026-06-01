// backends/cpu/kernels/signal/spectral/spectrogram_accum.cpp
// Accumulate real part from complex STFT output for spectrogram computation.
// Equivalent to: out[i] = real(in[i]) for each element.
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <complex>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status spectrogram_accum_kernel_cpu(void **inputs, void **outputs) {
  // inputs[0] = Xf (complex STFT output, n_freq × n_segments)
  // outputs[0] = Pxy_real (real part extracted, same shape)
  InsightArray *in = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!in || !out) {
    cpu_set_last_error("spectrogram_accum: null array pointer");
    return C_FAILED;
  }

  int64_t numel = in->numel;

  switch (in->dtype) {
  case INSIGHT_DTYPE_C64: {
    const std::complex<double> *src =
        reinterpret_cast<const std::complex<double> *>(in->data);
    double *dst = (double *)out->data;

#ifdef _OPENMP
    if (numel > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < numel; ++i) {
        dst[i] = src[i].real();
      }
    } else {
#endif
      for (int64_t i = 0; i < numel; ++i) {
        dst[i] = src[i].real();
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_C32: {
    const std::complex<float> *src =
        reinterpret_cast<const std::complex<float> *>(in->data);
    float *dst = (float *)out->data;

#ifdef _OPENMP
    if (numel > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < numel; ++i) {
        dst[i] = src[i].real();
      }
    } else {
#endif
      for (int64_t i = 0; i < numel; ++i) {
        dst[i] = src[i].real();
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  default:
    cpu_set_last_error(
        "spectrogram_accum: unsupported dtype, need C32 or C64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

// Also register as a power-spectrum variant: |Xf|^2 = real^2 + imag^2
C_Status spectrogram_power_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *in = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!in || !out) {
    cpu_set_last_error("spectrogram_power: null array pointer");
    return C_FAILED;
  }

  int64_t numel = in->numel;

  switch (in->dtype) {
  case INSIGHT_DTYPE_C64: {
    const std::complex<double> *src =
        reinterpret_cast<const std::complex<double> *>(in->data);
    double *dst = (double *)out->data;

#ifdef _OPENMP
    if (numel > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < numel; ++i) {
        dst[i] = std::norm(src[i]);
      }
    } else {
#endif
      for (int64_t i = 0; i < numel; ++i) {
        dst[i] = std::norm(src[i]);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_C32: {
    const std::complex<float> *src =
        reinterpret_cast<const std::complex<float> *>(in->data);
    float *dst = (float *)out->data;

#ifdef _OPENMP
    if (numel > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < numel; ++i) {
        dst[i] = std::norm(src[i]);
      }
    } else {
#endif
      for (int64_t i = 0; i < numel; ++i) {
        dst[i] = std::norm(src[i]);
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  default:
    cpu_set_last_error(
        "spectrogram_power: unsupported dtype, need C32 or C64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(spectrogram_accum, INSIGHT_DTYPE_C64,
                   spectrogram_accum_kernel_cpu);
REGISTER_CPU_KERNEL(spectrogram_accum, INSIGHT_DTYPE_C32,
                   spectrogram_accum_kernel_cpu);
REGISTER_CPU_KERNEL(spectrogram_power, INSIGHT_DTYPE_C64,
                   spectrogram_power_kernel_cpu);
REGISTER_CPU_KERNEL(spectrogram_power, INSIGHT_DTYPE_C32,
                   spectrogram_power_kernel_cpu);
