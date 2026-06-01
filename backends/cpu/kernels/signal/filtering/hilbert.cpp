// backends/cpu/kernels/signal/filtering/hilbert.cpp
// CPU kernel for Hilbert transform phase shifting.
// For each frequency bin: negative freq -> multiply by -j, positive -> by j,
// DC and Nyquist unchanged. This constructs the analytic signal in frequency
// domain.
// inputs[0]: complex spectrum array (C64 or C32)
// inputs[1]: n (signal length, I64, 1-element)
// outputs[0]: modified spectrum (same type)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <complex>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status signal_hilbert_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *spec_arr = (InsightArray *)inputs[0];
  InsightArray *n_arr = (InsightArray *)inputs[1];
  InsightArray *out_arr = (InsightArray *)outputs[0];

  if (!spec_arr || !n_arr || !out_arr) {
    cpu_set_last_error("signal_hilbert: null array pointer");
    return C_FAILED;
  }

  if (!spec_arr->data || !n_arr->data || !out_arr->data) {
    cpu_set_last_error("signal_hilbert: null data pointer");
    return C_FAILED;
  }

  int64_t n = *(int64_t *)n_arr->data;
  int64_t numel = spec_arr->numel;
  bool even = (n % 2 == 0);
  int64_t nfft = even ? (n / 2 + 1) : ((n + 1) / 2);

  switch (spec_arr->dtype) {
  case INSIGHT_DTYPE_C64: {
    const std::complex<double> *src =
        reinterpret_cast<const std::complex<double> *>(spec_arr->data);
    std::complex<double> *dst =
        reinterpret_cast<std::complex<double> *>(out_arr->data);
    const std::complex<double> j_val(0.0, 1.0);
    const std::complex<double> neg_j_val(0.0, -1.0);

#ifdef _OPENMP
    if (numel > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < numel; ++i) {
        int64_t k = i % nfft; // frequency bin index (for batched input)
        if (k == 0) {
          dst[i] = src[i]; // DC unchanged
        } else if (even && k == nfft - 1) {
          dst[i] = src[i]; // Nyquist unchanged (even n)
        } else {
          dst[i] = src[i] * j_val; // positive freq: multiply by j
        }
      }
    } else {
#endif
      for (int64_t i = 0; i < numel; ++i) {
        int64_t k = i % nfft;
        if (k == 0) {
          dst[i] = src[i];
        } else if (even && k == nfft - 1) {
          dst[i] = src[i];
        } else {
          dst[i] = src[i] * j_val;
        }
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_C32: {
    const std::complex<float> *src =
        reinterpret_cast<const std::complex<float> *>(spec_arr->data);
    std::complex<float> *dst =
        reinterpret_cast<std::complex<float> *>(out_arr->data);
    const std::complex<float> j_val(0.0f, 1.0f);

#ifdef _OPENMP
    if (numel > 1000) {
#pragma omp parallel for
      for (int64_t i = 0; i < numel; ++i) {
        int64_t k = i % nfft;
        if (k == 0) {
          dst[i] = src[i];
        } else if (even && k == nfft - 1) {
          dst[i] = src[i];
        } else {
          dst[i] = src[i] * j_val;
        }
      }
    } else {
#endif
      for (int64_t i = 0; i < numel; ++i) {
        int64_t k = i % nfft;
        if (k == 0) {
          dst[i] = src[i];
        } else if (even && k == nfft - 1) {
          dst[i] = src[i];
        } else {
          dst[i] = src[i] * j_val;
        }
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  default:
    cpu_set_last_error("signal_hilbert: unsupported dtype, need C32 or C64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_hilbert, INSIGHT_DTYPE_C64,
                    signal_hilbert_kernel_cpu);
REGISTER_CPU_KERNEL(signal_hilbert, INSIGHT_DTYPE_C32,
                    signal_hilbert_kernel_cpu);
