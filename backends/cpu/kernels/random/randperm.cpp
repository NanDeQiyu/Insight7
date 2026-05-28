// backends/cpu/kernels/random/randperm.cpp
/**
 * @file randperm.cpp
 * @brief CPU kernel for random permutation.
 *
 * Generates a random permutation of [0, n-1] using Fisher-Yates shuffle.
 *
 * @param inputs  [0] = InsightArray* output array
 * @param outputs [0] = InsightArray* output array
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status randperm_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("randperm: output array is null");
    return C_FAILED;
  }

  // randperm only works on contiguous arrays (1D)
  if (!insight_array_is_contiguous(out)) {
    cpu_set_last_error("randperm: output array must be contiguous");
    return C_FAILED;
  }

  switch (out->dtype) {
  case INSIGHT_DTYPE_I32:
    RANDPERM_LOOP(int32_t);
    break;
  case INSIGHT_DTYPE_I64:
    RANDPERM_LOOP(int64_t);
    break;
  default:
    cpu_set_last_error("randperm: unsupported dtype (need int32/int64)");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(randperm, INSIGHT_DTYPE_I32, randperm_kernel_cpu);
REGISTER_CPU_KERNEL(randperm, INSIGHT_DTYPE_I64, randperm_kernel_cpu);
