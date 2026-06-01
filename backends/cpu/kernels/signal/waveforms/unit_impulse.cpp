// backends/cpu/kernels/signal/waveforms/unit_impulse.cpp
// CPU kernel for unit impulse (Kronecker delta)
// w[n] = 1 if n == idx else 0
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

// unit_impulse kernel
// inputs: [0]=idx (1-element I64, flat index of impulse)
// outputs: [0]=impulse array (F64)
C_Status unit_impulse_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *idx_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!idx_arr || !out) {
    cpu_set_last_error("unit_impulse: null array pointer");
    return C_FAILED;
  }

  if (!idx_arr->data || idx_arr->numel < 1) {
    cpu_set_last_error("unit_impulse: idx array is empty");
    return C_FAILED;
  }

  int64_t idx = *(const int64_t *)idx_arr->data;
  int64_t M = out->numel;

  if (idx < 0 || idx >= M) {
    cpu_set_last_error("unit_impulse: idx out of range");
    return C_FAILED;
  }

  // Zero-fill then set impulse
  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    std::memset(data, 0, M * sizeof(double));
    data[idx] = 1.0;
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    std::memset(data, 0, M * sizeof(float));
    data[idx] = 1.0f;
  } else {
    cpu_set_last_error("unit_impulse: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(unit_impulse, INSIGHT_DTYPE_F64, unit_impulse_kernel_cpu);
REGISTER_CPU_KERNEL(unit_impulse, INSIGHT_DTYPE_F32, unit_impulse_kernel_cpu);
