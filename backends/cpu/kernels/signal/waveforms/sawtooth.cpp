// backends/cpu/kernels/signal/waveforms/sawtooth.cpp
// CPU kernel for sawtooth/triangle waveform
// w[n] = 2 * (t*n - floor(0.5 + t*n)) where t = width parameter
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

// sawtooth kernel
// inputs: [0]=width (1-element F64, default 1.0)
// outputs: [0]=waveform (F64)
C_Status sawtooth_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *width_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!width_arr || !out) {
    cpu_set_last_error("sawtooth: null array pointer");
    return C_FAILED;
  }

  if (!width_arr->data || width_arr->numel < 1) {
    cpu_set_last_error("sawtooth: width array is empty");
    return C_FAILED;
  }

  double width = *(const double *)width_arr->data;
  int64_t M = out->numel;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      double tn = width * (double)i;
      data[i] = 2.0 * (tn - std::floor(0.5 + tn));
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    float wf = (float)width;
#pragma omp parallel for if (M > 1000)
    for (int64_t i = 0; i < M; ++i) {
      float tn = wf * (float)i;
      data[i] = 2.0f * (tn - std::floor(0.5f + tn));
    }
  } else {
    cpu_set_last_error("sawtooth: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(sawtooth, INSIGHT_DTYPE_F64, sawtooth_kernel_cpu);
REGISTER_CPU_KERNEL(sawtooth, INSIGHT_DTYPE_F32, sawtooth_kernel_cpu);
