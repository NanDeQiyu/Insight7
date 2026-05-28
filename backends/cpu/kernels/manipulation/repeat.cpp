// backends/cpu/kernels/manipulation/repeat.cpp
/**
 * @file repeat.cpp
 * @brief CPU kernel for repeat operation.
 *
 * Repeats each element along the specified axis.
 * Supports non-contiguous arrays via stride-based indexing.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (unused, outputs[0] is used instead)
 *   inputs[1] = InsightArray* input
 *   inputs[2] = int* repeats
 *   inputs[3] = int* axis
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <complex>

// ============================================================================
// Repeat helper template
// ============================================================================

/**
 * @brief Repeat elements along an axis with stride support.
 *
 * @tparam T Data type
 * @param src_arr Source array (with possible offset and non-contiguous strides)
 * @param dst_arr Destination array (with possible offset)
 * @param ndim Number of dimensions
 * @param out_dims Output dimension sizes
 * @param axis Axis to repeat
 * @param repeats Number of repetitions per element
 * @return C_SUCCESS on success, C_FAILED on error
 */
template <typename T>
static C_Status cpu_repeat_impl(InsightArray *src_arr, InsightArray *dst_arr,
                                int64_t ndim, const int64_t *out_dims,
                                int64_t axis, int64_t repeats) {

  // Get source array properties
  const T *src = (const T *)src_arr->data;
  int64_t src_offset = src_arr->offset;
  int64_t src_strides[INSIGHT_MAX_NDIM];
  int64_t src_dims[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    src_strides[i] = src_arr->strides[i];
    src_dims[i] = src_arr->dims[i];
  }

  // Get destination array properties
  T *dst = (T *)dst_arr->data;
  int64_t dst_offset = dst_arr->offset;
  int64_t dst_strides[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    dst_strides[i] = dst_arr->strides[i];
  }

  int64_t total = 1;
  for (int64_t d = 0; d < ndim; ++d) {
    total *= out_dims[d];
  }

#pragma omp parallel for
  for (int64_t linear = 0; linear < total; ++linear) {
    int64_t indices[INSIGHT_MAX_NDIM];
    cpu_linear_to_indices(linear, ndim, out_dims, indices);

    // Map output index to source index along the repeat axis
    int64_t src_idx = indices[axis] / repeats;

    // Compute source offset (including view offset)
    int64_t src_off = src_offset;
    for (int64_t d = 0; d < ndim; ++d) {
      int64_t idx = (d == axis) ? src_idx : indices[d];
      src_off += idx * src_strides[d];
    }

    // Compute destination offset (including view offset)
    int64_t dst_off = dst_offset;
    for (int64_t d = 0; d < ndim; ++d) {
      dst_off += indices[d] * dst_strides[d];
    }

    dst[dst_off] = src[src_off];
  }

  return C_SUCCESS;
}

// ============================================================================
// Type dispatch macro
// ============================================================================

#define REPEAT_CASE(DTYPE, CTYPE)                                              \
  case DTYPE:                                                                  \
    return cpu_repeat_impl<CTYPE>(x, out, ndim, out_dims, axis, repeats);

// ============================================================================
// Kernel entry point
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

C_Status repeat_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("repeat: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2] || !inputs[3]) {
    cpu_set_last_error("repeat: repeats or axis is null");
    return C_FAILED;
  }

  int repeats = *(int *)inputs[2];
  int axis = *(int *)inputs[3];
  int64_t ndim = x->ndim;

  // Normalize axis
  if (axis < 0)
    axis += ndim;
  if (axis < 0 || axis >= ndim) {
    cpu_set_last_error("repeat: axis out of range");
    return C_FAILED;
  }

  if (repeats < 1) {
    cpu_set_last_error("repeat: repeats must be >= 1");
    return C_FAILED;
  }

  // Compute output dimensions
  int64_t out_dims[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    out_dims[i] = (i == axis) ? x->dims[i] * repeats : x->dims[i];
  }

  // Note: Output is allocated by frontend, may be non-contiguous
  // We rely on strides stored in out array

  switch (x->dtype) {
    REPEAT_CASE(INSIGHT_DTYPE_BOOL, bool)
    REPEAT_CASE(INSIGHT_DTYPE_U8, uint8_t)
    REPEAT_CASE(INSIGHT_DTYPE_I8, int8_t)
    REPEAT_CASE(INSIGHT_DTYPE_I16, int16_t)
    REPEAT_CASE(INSIGHT_DTYPE_I32, int32_t)
    REPEAT_CASE(INSIGHT_DTYPE_I64, int64_t)
    REPEAT_CASE(INSIGHT_DTYPE_U16, uint16_t)
    REPEAT_CASE(INSIGHT_DTYPE_U32, uint32_t)
    REPEAT_CASE(INSIGHT_DTYPE_U64, uint64_t)
    REPEAT_CASE(INSIGHT_DTYPE_F16, uint16_t)
    REPEAT_CASE(INSIGHT_DTYPE_BF16, uint16_t)
    REPEAT_CASE(INSIGHT_DTYPE_F32, float)
    REPEAT_CASE(INSIGHT_DTYPE_F64, double)
    REPEAT_CASE(INSIGHT_DTYPE_C32, std::complex<float>)
    REPEAT_CASE(INSIGHT_DTYPE_C64, std::complex<double>)
  default:
    cpu_set_last_error("repeat: unsupported dtype");
    return C_FAILED;
  }
}

#ifdef __cplusplus
}
#endif

// ============================================================================
// Kernel registration for all supported types
// ============================================================================

REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_BOOL, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_U8, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_I8, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_I16, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_I32, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_I64, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_U16, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_U32, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_U64, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_F16, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_BF16, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_F32, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_F64, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_C32, repeat_kernel_cpu);
REGISTER_CPU_KERNEL(repeat, INSIGHT_DTYPE_C64, repeat_kernel_cpu);