// backends/cpu/kernels/signal/io/unpack_bin.cpp
// CPU kernel for unpacking bytes to typed array.
// Reinterprets uint8 bytes as target dtype via memcpy.
// inputs[0]: byte array (U8)
// outputs[0]: typed array (any dtype, pre-allocated with correct size)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cstring>

extern "C" {

C_Status signal_unpack_bin_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *in_arr = (InsightArray *)inputs[0];
  InsightArray *out_arr = (InsightArray *)outputs[0];

  if (!in_arr || !out_arr) {
    cpu_set_last_error("signal_unpack_bin: null array pointer");
    return C_FAILED;
  }

  if (!in_arr->data || !out_arr->data) {
    cpu_set_last_error("signal_unpack_bin: null data pointer");
    return C_FAILED;
  }

  size_t nbytes = (size_t)(out_arr->numel * insight_dtype_size(out_arr->dtype));
  // Clamp to available input bytes to avoid overrun
  size_t in_nbytes = (size_t)in_arr->numel;
  if (nbytes > in_nbytes)
    nbytes = in_nbytes;

  std::memcpy(out_arr->data, in_arr->data, nbytes);

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_unpack_bin, INSIGHT_DTYPE_U8,
                    signal_unpack_bin_kernel_cpu);
