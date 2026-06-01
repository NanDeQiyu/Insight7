// backends/cpu/kernels/reduction/cumsum.cpp
/**
 * @file cumsum.cpp
 * @brief CPU kernel for cumulative sum operation.
 *
 * Computes cumulative sum along specified axis.
 *
 * @param inputs  [0] = InsightArray* output
 *                [1] = InsightArray* input
 *                [2] = int* axis
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <vector>

// ============================================================================
// Helper: compute outer and inner strides
// ============================================================================

static void compute_strides(int64_t ndim, const int64_t *dims, int axis,
                            int64_t *outer_stride, int64_t *inner_stride) {
  *outer_stride = 1;
  for (int i = 0; i < axis; ++i) {
    *outer_stride *= dims[i];
  }
  *inner_stride = 1;
  for (int i = axis + 1; i < ndim; ++i) {
    *inner_stride *= dims[i];
  }
}

// ============================================================================
// Template implementation
// ============================================================================

template <typename T>
static void cumsum_impl(const T *src, T *dst, int64_t axis_size,
                        int64_t outer_stride, int64_t inner_stride) {
#pragma omp parallel for collapse(2)
  for (int64_t outer = 0; outer < outer_stride; ++outer) {
    for (int64_t inner = 0; inner < inner_stride; ++inner) {
      T acc = 0;
      for (int64_t k = 0; k < axis_size; ++k) {
        int64_t idx =
            outer * axis_size * inner_stride + k * inner_stride + inner;
        acc += src[idx];
        dst[idx] = acc;
      }
    }
  }
}

// ============================================================================
// Kernel entry point
// ============================================================================
#ifdef __cplusplus
extern "C" {
#endif

C_Status cumsum_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("cumsum: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2]) {
    cpu_set_last_error("cumsum: axis is null");
    return C_FAILED;
  }

  int axis = *(int *)inputs[2];
  int64_t ndim = x->ndim;

  // Normalize axis
  if (axis < 0)
    axis += ndim;
  if (axis < 0 || axis >= ndim) {
    cpu_set_last_error("cumsum: axis out of range");
    return C_FAILED;
  }

  // Copy dimensions
  int64_t dims[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    dims[i] = x->dims[i];
  }

  int64_t axis_size = dims[axis];
  int64_t outer_stride, inner_stride;
  compute_strides(ndim, dims, axis, &outer_stride, &inner_stride);

  // Print first few input values
  const float *f32_src = (const float *)x->data;

  // Ensure contiguous (should be by frontend)
  if (!insight_array_is_contiguous(x)) {
    cpu_set_last_error("cumsum: input must be contiguous");
    return C_FAILED;
  }

  // Dispatch based on dtype
  switch (x->dtype) {
  case INSIGHT_DTYPE_F32:
    cumsum_impl<float>((const float *)x->data, (float *)out->data, axis_size,
                       outer_stride, inner_stride);
    break;
  case INSIGHT_DTYPE_F64:
    cumsum_impl<double>((const double *)x->data, (double *)out->data, axis_size,
                        outer_stride, inner_stride);
    break;
  case INSIGHT_DTYPE_I32:
    cumsum_impl<int32_t>((const int32_t *)x->data, (int32_t *)out->data,
                         axis_size, outer_stride, inner_stride);
    break;
  case INSIGHT_DTYPE_I64:
    cumsum_impl<int64_t>((const int64_t *)x->data, (int64_t *)out->data,
                         axis_size, outer_stride, inner_stride);
    break;
  case INSIGHT_DTYPE_F16: {
    const uint16_t *src = (const uint16_t *)x->data;
    uint16_t *dst = (uint16_t *)out->data;
#pragma omp parallel for collapse(2)
    for (int64_t outer = 0; outer < outer_stride; ++outer) {
      for (int64_t inner = 0; inner < inner_stride; ++inner) {
        float acc = 0.0f;
        for (int64_t k = 0; k < axis_size; ++k) {
          int64_t idx =
              outer * axis_size * inner_stride + k * inner_stride + inner;
          acc += insight::f16_to_f32(src[idx]);
          dst[idx] = insight::f32_to_f16(acc);
        }
      }
    }
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    const uint16_t *src = (const uint16_t *)x->data;
    uint16_t *dst = (uint16_t *)out->data;
#pragma omp parallel for collapse(2)
    for (int64_t outer = 0; outer < outer_stride; ++outer) {
      for (int64_t inner = 0; inner < inner_stride; ++inner) {
        float acc = 0.0f;
        for (int64_t k = 0; k < axis_size; ++k) {
          int64_t idx =
              outer * axis_size * inner_stride + k * inner_stride + inner;
          acc += insight::bf16_to_f32(src[idx]);
          dst[idx] = insight::f32_to_bf16(acc);
        }
      }
    }
    break;
  }
  default:
    cpu_set_last_error("cumsum: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(cumsum, INSIGHT_DTYPE_F32, cumsum_kernel_cpu);
REGISTER_CPU_KERNEL(cumsum, INSIGHT_DTYPE_F64, cumsum_kernel_cpu);
REGISTER_CPU_KERNEL(cumsum, INSIGHT_DTYPE_I32, cumsum_kernel_cpu);
REGISTER_CPU_KERNEL(cumsum, INSIGHT_DTYPE_I64, cumsum_kernel_cpu);