// backends/cpu/kernels/signal/windows/boxcar.cpp
// CPU kernel for boxcar (rectangular) window generation.
// w[n] = 1.0 for all n
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cstring>

extern "C" {

C_Status boxcar_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("boxcar: null output pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    for (int64_t i = 0; i < M; ++i) {
      data[i] = 1.0;
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    for (int64_t i = 0; i < M; ++i) {
      data[i] = 1.0f;
    }
  } else {
    cpu_set_last_error("boxcar: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(boxcar, INSIGHT_DTYPE_F64, boxcar_kernel_cpu);
REGISTER_CPU_KERNEL(boxcar, INSIGHT_DTYPE_F32, boxcar_kernel_cpu);
