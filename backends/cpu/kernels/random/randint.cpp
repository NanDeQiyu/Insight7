// backends/cpu/kernels/random/randint.cpp
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status randint_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("randint: output array is null");
    return C_FAILED;
  }

  int64_t low = *(int64_t *)inputs[1];
  int64_t high = *(int64_t *)inputs[2];

  switch (out->dtype) {
  case INSIGHT_DTYPE_I32:
    RANDOM_FILL_LOOP(int32_t, std::uniform_int_distribution<int32_t>,
                     (low, high - 1));
    break;
  case INSIGHT_DTYPE_I64:
    RANDOM_FILL_LOOP(int64_t, std::uniform_int_distribution<int64_t>,
                     (low, high - 1));
    break;
  default:
    cpu_set_last_error("randint: unsupported dtype (need int32/int64)");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(randint, INSIGHT_DTYPE_I32, randint_kernel_cpu);
REGISTER_CPU_KERNEL(randint, INSIGHT_DTYPE_I64, randint_kernel_cpu);