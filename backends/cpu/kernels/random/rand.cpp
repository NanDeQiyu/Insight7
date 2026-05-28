// backends/cpu/kernels/random/rand.cpp
/**
 * @file rand.cpp
 * @brief CPU kernel for Uniform distribution.
 *
 * Generates random numbers from uniform_real_distribution distribution.
 * Supports both contiguous and strided arrays.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = uint64_t* device seed (unused, CPU uses global seed)
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status rand_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("rand: output array is null");
    return C_FAILED;
  }

  // Seed is ignored for CPU backend (uses global seed)
  // uint64_t device_seed = *(uint64_t*)inputs[1];

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    RANDOM_FILL_LOOP(float, std::uniform_real_distribution<float>, (0.0, 1.0));
    break;
  case INSIGHT_DTYPE_F64:
    RANDOM_FILL_LOOP(double, std::uniform_real_distribution<double>,
                     (0.0, 1.0));
    break;
  default:
    cpu_set_last_error("rand: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(rand, INSIGHT_DTYPE_F32, rand_kernel_cpu);
REGISTER_CPU_KERNEL(rand, INSIGHT_DTYPE_F64, rand_kernel_cpu);
