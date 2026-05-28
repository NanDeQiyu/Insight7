// backends/cpu/kernels/manipulation/roll.cpp
/**
 * @file roll.cpp
 * @brief CPU kernel for roll operation.
 *
 * Rolls elements along the specified axis.
 *
 * @param inputs  [0] = InsightArray* output
 *                [1] = InsightArray* input
 *                [2] = int* shift
 *                [3] = int* axis
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

// ============================================================================
// Roll helper (1D)
// ============================================================================

/**
 * @brief Roll a contiguous 1D array.
 *
 * @tparam T Data type
 * @param src Source data
 * @param dst Destination data
 * @param n Number of elements
 * @param shift Shift amount
 */
template <typename T>
static void cpu_roll_1d_impl(const T *src, T *dst, int64_t n, int64_t shift) {

  shift = ((shift % n) + n) % n;
  if (shift == 0) {
    memcpy(dst, src, n * sizeof(T));
    return;
  }

#pragma omp parallel for
  for (int64_t i = 0; i < n; ++i) {
    int64_t src_idx = (i - shift + n) % n;
    dst[i] = src[src_idx];
  }
}

// ============================================================================
// Roll helper (ND)
// ============================================================================

/**
 * @brief Roll an ND array along an axis.
 *
 * @tparam T Data type
 * @param src Source data
 * @param dst Destination data
 * @param ndim Number of dimensions
 * @param dims Dimension sizes
 * @param strides Strides
 * @param axis Axis to roll
 * @param shift Shift amount
 */
template <typename T>
static void cpu_roll_nd_impl(const T *src, T *dst, int64_t ndim,
                             const int64_t *dims, const int64_t *strides,
                             int64_t axis, int64_t shift) {

  int64_t axis_size = dims[axis];
  int64_t axis_stride = strides[axis];
  shift = ((shift % axis_size) + axis_size) % axis_size;

  if (shift == 0) {
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
      total *= dims[d];
    }
    memcpy(dst, src, total * sizeof(T));
    return;
  }

  int64_t outer_stride = 1;
  for (int64_t d = 0; d < axis; ++d) {
    outer_stride *= dims[d];
  }

  int64_t inner_stride = 1;
  for (int64_t d = axis + 1; d < ndim; ++d) {
    inner_stride *= dims[d];
  }

  int64_t block_size = axis_size * inner_stride;
  int64_t num_blocks = outer_stride;

#pragma omp parallel for collapse(2)
  for (int64_t block = 0; block < num_blocks; ++block) {
    for (int64_t inner = 0; inner < inner_stride; ++inner) {
      for (int64_t k = 0; k < axis_size; ++k) {
        int64_t src_idx = block * block_size + k * inner_stride + inner;
        int64_t dst_idx = block * block_size +
                          ((k + shift) % axis_size) * inner_stride + inner;
        dst[dst_idx] = src[src_idx];
      }
    }
  }
}

#ifdef __cplusplus
extern "C" {
#endif

C_Status roll_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("roll: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2] || !inputs[3]) {
    cpu_set_last_error("roll: shift or axis is null");
    return C_FAILED;
  }

  int shift = *(int *)inputs[2];
  int axis = *(int *)inputs[3];
  int64_t ndim = x->ndim;

  // Ensure output is contiguous (should be by frontend)
  if (!insight_array_is_contiguous(out)) {
    cpu_set_last_error("roll: output must be contiguous");
    return C_FAILED;
  }

  if (axis == -1) {
    // Flatten case
    int64_t n = x->numel;
    switch (x->dtype) {
    case INSIGHT_DTYPE_BOOL:
      cpu_roll_1d_impl<bool>((const bool *)x->data, (bool *)out->data, n,
                             shift);
      break;
    case INSIGHT_DTYPE_U8:
      cpu_roll_1d_impl<uint8_t>((const uint8_t *)x->data, (uint8_t *)out->data,
                                n, shift);
      break;
    case INSIGHT_DTYPE_I8:
      cpu_roll_1d_impl<int8_t>((const int8_t *)x->data, (int8_t *)out->data, n,
                               shift);
      break;
    case INSIGHT_DTYPE_I16:
      cpu_roll_1d_impl<int16_t>((const int16_t *)x->data, (int16_t *)out->data,
                                n, shift);
      break;
    case INSIGHT_DTYPE_I32:
      cpu_roll_1d_impl<int32_t>((const int32_t *)x->data, (int32_t *)out->data,
                                n, shift);
      break;
    case INSIGHT_DTYPE_I64:
      cpu_roll_1d_impl<int64_t>((const int64_t *)x->data, (int64_t *)out->data,
                                n, shift);
      break;
    case INSIGHT_DTYPE_U16:
      cpu_roll_1d_impl<uint16_t>((const uint16_t *)x->data,
                                 (uint16_t *)out->data, n, shift);
      break;
    case INSIGHT_DTYPE_U32:
      cpu_roll_1d_impl<uint32_t>((const uint32_t *)x->data,
                                 (uint32_t *)out->data, n, shift);
      break;
    case INSIGHT_DTYPE_U64:
      cpu_roll_1d_impl<uint64_t>((const uint64_t *)x->data,
                                 (uint64_t *)out->data, n, shift);
      break;
    case INSIGHT_DTYPE_F32:
      cpu_roll_1d_impl<float>((const float *)x->data, (float *)out->data, n,
                              shift);
      break;
    case INSIGHT_DTYPE_F64:
      cpu_roll_1d_impl<double>((const double *)x->data, (double *)out->data, n,
                               shift);
      break;
    default:
      cpu_set_last_error("roll: unsupported dtype");
      return C_FAILED;
    }
  } else {
    // ND case
    if (axis < 0)
      axis += ndim;
    if (axis < 0 || axis >= ndim) {
      cpu_set_last_error("roll: axis out of range");
      return C_FAILED;
    }

    int64_t dims[INSIGHT_MAX_NDIM];
    int64_t strides[INSIGHT_MAX_NDIM];
    for (int i = 0; i < ndim; ++i) {
      dims[i] = x->dims[i];
      strides[i] = x->strides[i];
    }

    switch (x->dtype) {
    case INSIGHT_DTYPE_BOOL:
      cpu_roll_nd_impl<bool>((const bool *)x->data, (bool *)out->data, ndim,
                             dims, strides, axis, shift);
      break;
    case INSIGHT_DTYPE_U8:
      cpu_roll_nd_impl<uint8_t>((const uint8_t *)x->data, (uint8_t *)out->data,
                                ndim, dims, strides, axis, shift);
      break;
    case INSIGHT_DTYPE_I8:
      cpu_roll_nd_impl<int8_t>((const int8_t *)x->data, (int8_t *)out->data,
                               ndim, dims, strides, axis, shift);
      break;
    case INSIGHT_DTYPE_I16:
      cpu_roll_nd_impl<int16_t>((const int16_t *)x->data, (int16_t *)out->data,
                                ndim, dims, strides, axis, shift);
      break;
    case INSIGHT_DTYPE_I32:
      cpu_roll_nd_impl<int32_t>((const int32_t *)x->data, (int32_t *)out->data,
                                ndim, dims, strides, axis, shift);
      break;
    case INSIGHT_DTYPE_I64:
      cpu_roll_nd_impl<int64_t>((const int64_t *)x->data, (int64_t *)out->data,
                                ndim, dims, strides, axis, shift);
      break;
    case INSIGHT_DTYPE_U16:
      cpu_roll_nd_impl<uint16_t>((const uint16_t *)x->data,
                                 (uint16_t *)out->data, ndim, dims, strides,
                                 axis, shift);
      break;
    case INSIGHT_DTYPE_U32:
      cpu_roll_nd_impl<uint32_t>((const uint32_t *)x->data,
                                 (uint32_t *)out->data, ndim, dims, strides,
                                 axis, shift);
      break;
    case INSIGHT_DTYPE_U64:
      cpu_roll_nd_impl<uint64_t>((const uint64_t *)x->data,
                                 (uint64_t *)out->data, ndim, dims, strides,
                                 axis, shift);
      break;
    case INSIGHT_DTYPE_F32:
      cpu_roll_nd_impl<float>((const float *)x->data, (float *)out->data, ndim,
                              dims, strides, axis, shift);
      break;
    case INSIGHT_DTYPE_F64:
      cpu_roll_nd_impl<double>((const double *)x->data, (double *)out->data,
                               ndim, dims, strides, axis, shift);
      break;
    default:
      cpu_set_last_error("roll: unsupported dtype");
      return C_FAILED;
    }
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(roll, INSIGHT_DTYPE_BOOL, roll_kernel_cpu);
REGISTER_CPU_KERNEL(roll, INSIGHT_DTYPE_U8, roll_kernel_cpu);
REGISTER_CPU_KERNEL(roll, INSIGHT_DTYPE_I8, roll_kernel_cpu);
REGISTER_CPU_KERNEL(roll, INSIGHT_DTYPE_I16, roll_kernel_cpu);
REGISTER_CPU_KERNEL(roll, INSIGHT_DTYPE_I32, roll_kernel_cpu);
REGISTER_CPU_KERNEL(roll, INSIGHT_DTYPE_I64, roll_kernel_cpu);
REGISTER_CPU_KERNEL(roll, INSIGHT_DTYPE_U16, roll_kernel_cpu);
REGISTER_CPU_KERNEL(roll, INSIGHT_DTYPE_U32, roll_kernel_cpu);
REGISTER_CPU_KERNEL(roll, INSIGHT_DTYPE_U64, roll_kernel_cpu);
REGISTER_CPU_KERNEL(roll, INSIGHT_DTYPE_F32, roll_kernel_cpu);
REGISTER_CPU_KERNEL(roll, INSIGHT_DTYPE_F64, roll_kernel_cpu);
