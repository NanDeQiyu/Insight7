// backends/cpu/kernels/random/get_seed.cpp
/**
 * @file get_seed.cpp
 * @brief CPU kernel for getting random seed.
 *
 * Returns the current global seed for CPU backend.
 *
 * @param inputs  [0] = unused
 * @param outputs [0] = uint64_t* output seed
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status get_seed_kernel_cpu(void **inputs, void **outputs) {
  if (!outputs[0]) {
    cpu_set_last_error("get_seed: output pointer is null");
    return C_FAILED;
  }

  uint64_t *out = (uint64_t *)outputs[0];
  *out = cpu_get_global_seed();

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(get_seed, INSIGHT_DTYPE_U64, get_seed_kernel_cpu);
