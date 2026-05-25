// backends/cpu/kernels/reduction/nansum.cpp
/**
 * @file nansum.cpp
 * @brief CPU kernel for nansum reduction (sum ignoring NaN).
 *
 * Computes sum along reduction axis, skipping NaN values.
 * Outputs both sum and count (number of non-NaN elements).
 */

#include "common.h"
#include <cmath>
#include <cstdio>

// ============================================================================
// Template implementation (no extern C)
// ============================================================================

template <typename T>
static void nansum_loop(const T *src, T *sum_dst, int64_t *count_dst,
                        int64_t batch_size, int64_t reduce_size) {

#pragma omp parallel for
  for (int64_t i = 0; i < batch_size; ++i) {
    T sum = 0;
    int64_t cnt = 0;
    for (int64_t j = 0; j < reduce_size; ++j) {
      T val = src[i * reduce_size + j];
      if (!std::isnan(val)) {
        sum += val;
        ++cnt;
      }
    }
    sum_dst[i] = sum;
    count_dst[i] = cnt;
  }
}

// ============================================================================
// C API entry point
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

C_Status nansum_kernel_cpu(void **inputs, void **outputs) {
  // inputs:  [sum_out, count_out, prepared, batch_size, reduce_size]
  // outputs: [sum_out, count_out]
  InsightArray *sum_out = (InsightArray *)outputs[0];
  InsightArray *count_out = (InsightArray *)outputs[1];
  InsightArray *prepared = (InsightArray *)inputs[2];

  if (!sum_out || !count_out || !prepared) {
    cpu_set_last_error("nansum: null array pointer");
    return C_FAILED;
  }

  if (!inputs[3] || !inputs[4]) {
    cpu_set_last_error("nansum: batch_size or reduce_size is null");
    return C_FAILED;
  }

  int64_t batch_size = *(int64_t *)inputs[3];
  int64_t reduce_size = *(int64_t *)inputs[4];

  switch (prepared->dtype) {
  case INSIGHT_DTYPE_F32:
    nansum_loop<float>((const float *)prepared->data, (float *)sum_out->data,
                       (int64_t *)count_out->data, batch_size, reduce_size);
    break;
  case INSIGHT_DTYPE_F64:
    nansum_loop<double>((const double *)prepared->data, (double *)sum_out->data,
                        (int64_t *)count_out->data, batch_size, reduce_size);
    break;
  default:
    cpu_set_last_error("nansum: only float32 and float64 supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(nansum, INSIGHT_DTYPE_F32, nansum_kernel_cpu);
REGISTER_CPU_KERNEL(nansum, INSIGHT_DTYPE_F64, nansum_kernel_cpu);