// backends/cpu/kernels/manipulation/flip.cpp
/**
 * @file flip.cpp
 * @brief CPU kernel for flip operation.
 *
 * Reverses the order of elements along the specified axis.
 *
 * @param inputs  [0] = InsightArray* output
 *                [1] = InsightArray* input
 *                [2] = int* axis
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <complex>

// ============================================================================
// Flip helper
// ============================================================================

/**
 * @brief Flip an array along a specific axis.
 *
 * @tparam T Data type
 * @param src Source data
 * @param dst Destination data
 * @param ndim Number of dimensions
 * @param dims Dimension sizes
 * @param src_strides Source strides
 * @param dst_strides Destination strides (contiguous)
 * @param axis Axis to flip
 */
template <typename T>
static void cpu_flip_impl(const T *src, T *dst, int64_t ndim,
                          const int64_t *dims, const int64_t *src_strides,
                          const int64_t *dst_strides, int64_t axis) {

  int64_t axis_size = dims[axis];
  int64_t total = 1;
  for (int64_t d = 0; d < ndim; ++d) {
    total *= dims[d];
  }

#pragma omp parallel for
  for (int64_t linear = 0; linear < total; ++linear) {
    int64_t indices[INSIGHT_MAX_NDIM];
    cpu_linear_to_indices(linear, ndim, dims, indices);

    // Flip the target axis
    indices[axis] = axis_size - 1 - indices[axis];

    int64_t src_offset =
        cpu_indices_to_linear(ndim, dims, src_strides, indices);
    dst[linear] = src[src_offset];
  }
}

#ifdef __cplusplus
extern "C" {
#endif

C_Status flip_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("flip: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2]) {
    cpu_set_last_error("flip: axis is null");
    return C_FAILED;
  }

  int axis = *(int *)inputs[2];
  int64_t ndim = x->ndim;

  if (axis < 0)
    axis += ndim;
  if (axis < 0 || axis >= ndim) {
    cpu_set_last_error("flip: axis out of range");
    return C_FAILED;
  }

  int64_t dims[INSIGHT_MAX_NDIM];
  int64_t src_strides[INSIGHT_MAX_NDIM];
  int64_t dst_strides[INSIGHT_MAX_NDIM];

  for (int i = 0; i < ndim; ++i) {
    dims[i] = x->dims[i];
    src_strides[i] = x->strides[i];
  }

  cpu_compute_strides(ndim, dims, dst_strides);

  // Ensure output is contiguous (should be by frontend)
  if (!insight_array_is_contiguous(out)) {
    cpu_set_last_error("flip: output must be contiguous");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_BOOL:
    cpu_flip_impl<bool>((const bool *)x->data, (bool *)out->data, ndim, dims,
                        src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_U8:
    cpu_flip_impl<uint8_t>((const uint8_t *)x->data, (uint8_t *)out->data, ndim,
                           dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_I8:
    cpu_flip_impl<int8_t>((const int8_t *)x->data, (int8_t *)out->data, ndim,
                          dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_I16:
    cpu_flip_impl<int16_t>((const int16_t *)x->data, (int16_t *)out->data, ndim,
                           dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_I32:
    cpu_flip_impl<int32_t>((const int32_t *)x->data, (int32_t *)out->data, ndim,
                           dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_I64:
    cpu_flip_impl<int64_t>((const int64_t *)x->data, (int64_t *)out->data, ndim,
                           dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_U16:
    cpu_flip_impl<uint16_t>((const uint16_t *)x->data, (uint16_t *)out->data,
                            ndim, dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_U32:
    cpu_flip_impl<uint32_t>((const uint32_t *)x->data, (uint32_t *)out->data,
                            ndim, dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_U64:
    cpu_flip_impl<uint64_t>((const uint64_t *)x->data, (uint64_t *)out->data,
                            ndim, dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_F32:
    cpu_flip_impl<float>((const float *)x->data, (float *)out->data, ndim, dims,
                         src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_F64:
    cpu_flip_impl<double>((const double *)x->data, (double *)out->data, ndim,
                          dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_C32:
    cpu_flip_impl<std::complex<float>>((const std::complex<float> *)x->data,
                                       (std::complex<float> *)out->data, ndim,
                                       dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_C64:
    cpu_flip_impl<std::complex<double>>((const std::complex<double> *)x->data,
                                        (std::complex<double> *)out->data, ndim,
                                        dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_F16:
    cpu_flip_impl<uint16_t>((const uint16_t *)x->data, (uint16_t *)out->data,
                            ndim, dims, src_strides, dst_strides, axis);
    break;
  case INSIGHT_DTYPE_BF16:
    cpu_flip_impl<uint16_t>((const uint16_t *)x->data, (uint16_t *)out->data,
                            ndim, dims, src_strides, dst_strides, axis);
    break;
  default:
    cpu_set_last_error("flip: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_BOOL, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_U8, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_I8, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_I16, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_I32, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_I64, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_U16, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_U32, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_U64, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_F32, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_F64, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_C32, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_C64, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_F16, flip_kernel_cpu);
REGISTER_CPU_KERNEL(flip, INSIGHT_DTYPE_BF16, flip_kernel_cpu);
