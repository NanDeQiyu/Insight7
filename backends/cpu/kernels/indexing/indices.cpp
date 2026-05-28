// backends/cpu/kernels/indexing/indices.cpp
/**
 * @file indices.cpp
 * @brief CPU kernel for indices operation.
 *
 * Returns grid indices for given shape.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output
 *   inputs[1] = int* ndim
 *   inputs[2] = int64_t* dims
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status indices_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  int ndim = *(int *)inputs[1];
  int64_t *dims = (int64_t *)inputs[2];

  if (!out) {
    cpu_set_last_error("indices: null array pointer");
    return C_FAILED;
  }

  int64_t *dst = (int64_t *)out->data;
  int64_t total = 1;
  for (int i = 0; i < ndim; ++i) {
    total *= dims[i];
  }

  for (int d = 0; d < ndim; ++d) {
    int64_t *plane = dst + d * total;
    int64_t indices[INSIGHT_MAX_NDIM];
    for (int64_t linear = 0; linear < total; ++linear) {
      int64_t tmp = linear;
      for (int i = ndim - 1; i >= 0; --i) {
        indices[i] = tmp % dims[i];
        tmp /= dims[i];
      }
      plane[linear] = indices[d];
    }
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(indices, INSIGHT_DTYPE_I64, indices_kernel_cpu);
