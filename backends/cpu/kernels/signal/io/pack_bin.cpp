// backends/cpu/kernels/signal/io/pack_bin.cpp
// CPU kernel for packing array data to bytes.
// Reinterprets array data as uint8 bytes via memcpy.
// inputs[0]: data array (any dtype)
// outputs[0]: byte array (U8)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cstring>

extern "C" {

C_Status signal_pack_bin_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *in_arr = (InsightArray *)inputs[0];
  InsightArray *out_arr = (InsightArray *)outputs[0];

  if (!in_arr || !out_arr) {
    cpu_set_last_error("signal_pack_bin: null array pointer");
    return C_FAILED;
  }

  if (!in_arr->data || !out_arr->data) {
    cpu_set_last_error("signal_pack_bin: null data pointer");
    return C_FAILED;
  }

  size_t nbytes = (size_t)(in_arr->numel * insight_dtype_size(in_arr->dtype));
  std::memcpy(out_arr->data, in_arr->data, nbytes);

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_pack_bin, INSIGHT_DTYPE_U8,
                    signal_pack_bin_kernel_cpu);
