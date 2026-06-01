// NOTE: This kernel is currently not called from the frontend.
// The frontend uses composite ops for this operation.
// This kernel is preserved for future backend dispatch optimization.
//
// backends/cpu/kernels/signal/filtering/firfilter.cpp
// CPU kernel for FIR filter — applies convolution per batch row
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cstring>
#include <vector>

extern "C" {

// firfilter kernel
// inputs: [0]=x (batch_size x signal_len), [1]=b (filter coefficients)
// outputs: [0]=y (batch_size x out_len) where out_len = signal_len + nb - 1
C_Status firfilter_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *b_arr = (InsightArray *)inputs[1];
  InsightArray *y_arr = (InsightArray *)outputs[0];

  if (!x_arr || !b_arr || !y_arr) {
    cpu_set_last_error("firfilter: null array pointer");
    return C_FAILED;
  }

  int64_t batch = x_arr->dims[0];
  int64_t signal_len = x_arr->dims[1];
  int64_t nb = b_arr->numel;
  int64_t out_len = signal_len + nb - 1;

  if (x_arr->dtype == INSIGHT_DTYPE_F64) {
    const double *x_data = (const double *)x_arr->data;
    const double *b_data = (const double *)b_arr->data;
    double *out_data = (double *)y_arr->data;

    for (int64_t row = 0; row < batch; ++row) {
      const double *xi = x_data + row * signal_len;
      double *yi = out_data + row * out_len;
      for (int64_t i = 0; i < out_len; ++i) {
        double sum = 0.0;
        for (int64_t j = 0; j < nb; ++j) {
          int64_t idx = i - j;
          if (idx >= 0 && idx < signal_len) {
            sum += xi[idx] * b_data[j];
          }
        }
        yi[i] = sum;
      }
    }
  } else {
    const float *x_data = (const float *)x_arr->data;
    const float *b_data = (const float *)b_arr->data;
    float *out_data = (float *)y_arr->data;

    for (int64_t row = 0; row < batch; ++row) {
      const float *xi = x_data + row * signal_len;
      float *yi = out_data + row * out_len;
      for (int64_t i = 0; i < out_len; ++i) {
        float sum = 0.0f;
        for (int64_t j = 0; j < nb; ++j) {
          int64_t idx = i - j;
          if (idx >= 0 && idx < signal_len) {
            sum += xi[idx] * b_data[j];
          }
        }
        yi[i] = sum;
      }
    }
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(firfilter, INSIGHT_DTYPE_F64, firfilter_kernel_cpu);
REGISTER_CPU_KERNEL(firfilter, INSIGHT_DTYPE_F32, firfilter_kernel_cpu);
