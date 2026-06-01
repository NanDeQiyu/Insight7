// backends/cuda/kernels/signal/filtering/firfilter_zi_state.cu
// FIR filter with initial state: C_FALLBACK for now.
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"

extern "C" {

static C_Status firfilter_zi_state_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}

} // extern "C"

REGISTER_GPU_KERNEL(firfilter_zi_state, INSIGHT_DTYPE_F32,
                    firfilter_zi_state_kernel_gpu);
REGISTER_GPU_KERNEL(firfilter_zi_state, INSIGHT_DTYPE_F64,
                    firfilter_zi_state_kernel_gpu);
