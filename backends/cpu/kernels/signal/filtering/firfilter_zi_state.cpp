// backends/cpu/kernels/signal/filtering/firfilter_zi_state.cpp
// FIR filter with initial state support.
// Implements: y[i] = b[0]*x[i] + b[1]*x[i-1] + ... + b[nb-1]*x[i-nb+1]
// With initial state zi (length nb-1) prepended to x.
// Returns both the output y and the final state zf (last nb-1 samples of
// extended input).
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status firfilter_zi_state_kernel_cpu(void **inputs, void **outputs) {
  // inputs[0] = b (FIR coefficients, length nb)
  // inputs[1] = x (input signal, length n)
  // inputs[2] = zi (initial state, length nb-1)
  // outputs[0] = y (output, length n + nb - 1, full convolution)
  // outputs[1] = zf (final state, length nb-1, last nb-1 elements of extended
  // input)
  InsightArray *b_arr = (InsightArray *)inputs[0];
  InsightArray *x_arr = (InsightArray *)inputs[1];
  InsightArray *zi_arr = (InsightArray *)inputs[2];
  InsightArray *y_arr = (InsightArray *)outputs[0];
  InsightArray *zf_arr = (InsightArray *)outputs[1];

  if (!b_arr || !x_arr || !zi_arr || !y_arr || !zf_arr) {
    cpu_set_last_error("firfilter_zi_state: null array pointer");
    return C_FAILED;
  }

  int64_t nb = b_arr->numel; // number of filter taps
  int64_t n = x_arr->numel;  // input signal length
  int64_t zi_len = zi_arr->numel; // initial state length (should be nb-1)

  if (zi_len != nb - 1) {
    cpu_set_last_error(
        "firfilter_zi_state: zi length must be len(b) - 1");
    return C_FAILED;
  }

  // Build extended input: [zi, x] (length = nb-1 + n)
  int64_t n_ext = zi_len + n;

  switch (b_arr->dtype) {
  case INSIGHT_DTYPE_F64: {
    const double *b = (const double *)b_arr->data;
    const double *x = (const double *)x_arr->data;
    const double *zi = (const double *)zi_arr->data;
    double *y = (double *)y_arr->data;

    // Build extended input
    double *x_ext = new double[n_ext];
    std::memcpy(x_ext, zi, zi_len * sizeof(double));
    std::memcpy(x_ext + zi_len, x, n * sizeof(double));

    // Full convolution: y[k] = sum_{j=0}^{nb-1} b[j] * x_ext[k-j]
    // for k = 0 .. n_ext-1 (output length = n_ext)
    // But we only need k = zi_len-1 .. zi_len-1+n-1 = n_ext-1
    // i.e. output length = n + nb - 1
    int64_t out_len = y_arr->numel;

#ifdef _OPENMP
    if (out_len > 1000) {
#pragma omp parallel for
      for (int64_t k = 0; k < out_len; ++k) {
        double sum = 0.0;
        for (int64_t j = 0; j < nb; ++j) {
          int64_t idx = k - j;
          if (idx >= 0 && idx < n_ext) {
            sum += b[j] * x_ext[idx];
          }
        }
        y[k] = sum;
      }
    } else {
#endif
      for (int64_t k = 0; k < out_len; ++k) {
        double sum = 0.0;
        for (int64_t j = 0; j < nb; ++j) {
          int64_t idx = k - j;
          if (idx >= 0 && idx < n_ext) {
            sum += b[j] * x_ext[idx];
          }
        }
        y[k] = sum;
      }
#ifdef _OPENMP
    }
#endif

    // Final state: last nb-1 elements of extended input
    double *zf = (double *)zf_arr->data;
    std::memcpy(zf, x_ext + n_ext - zi_len, zi_len * sizeof(double));

    delete[] x_ext;
    break;
  }
  case INSIGHT_DTYPE_F32: {
    const float *b = (const float *)b_arr->data;
    const float *x = (const float *)x_arr->data;
    const float *zi = (const float *)zi_arr->data;
    float *y = (float *)y_arr->data;

    float *x_ext = new float[n_ext];
    std::memcpy(x_ext, zi, zi_len * sizeof(float));
    std::memcpy(x_ext + zi_len, x, n * sizeof(float));

    int64_t out_len = y_arr->numel;

#ifdef _OPENMP
    if (out_len > 1000) {
#pragma omp parallel for
      for (int64_t k = 0; k < out_len; ++k) {
        float sum = 0.0f;
        for (int64_t j = 0; j < nb; ++j) {
          int64_t idx = k - j;
          if (idx >= 0 && idx < n_ext) {
            sum += b[j] * x_ext[idx];
          }
        }
        y[k] = sum;
      }
    } else {
#endif
      for (int64_t k = 0; k < out_len; ++k) {
        float sum = 0.0f;
        for (int64_t j = 0; j < nb; ++j) {
          int64_t idx = k - j;
          if (idx >= 0 && idx < n_ext) {
            sum += b[j] * x_ext[idx];
          }
        }
        y[k] = sum;
      }
#ifdef _OPENMP
    }
#endif

    float *zf = (float *)zf_arr->data;
    std::memcpy(zf, x_ext + n_ext - zi_len, zi_len * sizeof(float));

    delete[] x_ext;
    break;
  }
  default:
    cpu_set_last_error(
        "firfilter_zi_state: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(firfilter_zi_state, INSIGHT_DTYPE_F32,
                   firfilter_zi_state_kernel_cpu);
REGISTER_CPU_KERNEL(firfilter_zi_state, INSIGHT_DTYPE_F64,
                   firfilter_zi_state_kernel_cpu);
