// backends/cpu/kernels/random/randn.cpp
/**
 * @file randn.cpp
 * @brief CPU kernel for Normal distribution.
 *
 * Generates random numbers from normal_distribution distribution.
 * Supports both contiguous and strided arrays.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = uint64_t* device seed (unused, CPU uses global seed)
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include "../common/half_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status randn_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("randn: output array is null");
    return C_FAILED;
  }

  // Seed is ignored for CPU backend (uses global seed)
  // uint64_t device_seed = *(uint64_t*)inputs[1];

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    RANDOM_FILL_LOOP(float, std::normal_distribution<float>, (0.0, 1.0));
    break;
  case INSIGHT_DTYPE_F64:
    RANDOM_FILL_LOOP(double, std::normal_distribution<double>, (0.0, 1.0));
    break;
  case INSIGHT_DTYPE_F16: {
    uint16_t *out_data = (uint16_t *)out->data;
    int64_t n = out->numel;
    int64_t ndim = out->ndim;
    int64_t dims[INSIGHT_MAX_NDIM];
    int64_t out_strides[INSIGHT_MAX_NDIM];
    for (int i = 0; i < ndim; ++i) {
      dims[i] = out->dims[i];
      out_strides[i] = out->strides[i];
    }
#pragma omp parallel
    {
      std::normal_distribution<float> dist(0.0f, 1.0f);
#pragma omp for
      for (int64_t linear = 0; linear < n; ++linear) {
        int64_t off_out =
            cpu_offset_from_linear(linear, ndim, dims, out_strides);
        out_data[off_out] = insight::f32_to_f16(dist(cpu_get_rng()));
      }
    }
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    uint16_t *out_data = (uint16_t *)out->data;
    int64_t n = out->numel;
    int64_t ndim = out->ndim;
    int64_t dims[INSIGHT_MAX_NDIM];
    int64_t out_strides[INSIGHT_MAX_NDIM];
    for (int i = 0; i < ndim; ++i) {
      dims[i] = out->dims[i];
      out_strides[i] = out->strides[i];
    }
#pragma omp parallel
    {
      std::normal_distribution<float> dist(0.0f, 1.0f);
#pragma omp for
      for (int64_t linear = 0; linear < n; ++linear) {
        int64_t off_out =
            cpu_offset_from_linear(linear, ndim, dims, out_strides);
        out_data[off_out] = insight::f32_to_bf16(dist(cpu_get_rng()));
      }
    }
    break;
  }
  default:
    cpu_set_last_error("randn: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(randn, INSIGHT_DTYPE_F32, randn_kernel_cpu);
REGISTER_CPU_KERNEL(randn, INSIGHT_DTYPE_F64, randn_kernel_cpu);
REGISTER_CPU_KERNEL(randn, INSIGHT_DTYPE_F16, randn_kernel_cpu);
REGISTER_CPU_KERNEL(randn, INSIGHT_DTYPE_BF16, randn_kernel_cpu);
