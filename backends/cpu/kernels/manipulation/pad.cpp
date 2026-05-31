// backends/cpu/kernels/manipulation/pad.cpp
/**
 * @file pad.cpp
 * @brief CPU kernel for pad operation.
 *
 * Pads an array with a constant value along each dimension.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output
 *   inputs[1] = InsightArray* input
 *   inputs[2] = int64_t* pad_width (2 * ndim values: before0, after0, before1,
 * after1, ...) inputs[3] = double* constant_value
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <complex>
#include <cstring>

// ============================================================================
// Helper: Compute output strides from output dimensions (row-major)
// ============================================================================

static inline void compute_out_strides(int64_t ndim, const int64_t *dims,
                                       int64_t *strides) {
  if (ndim == 0)
    return;
  strides[ndim - 1] = 1;
  for (int64_t d = ndim - 2; d >= 0; --d) {
    strides[d] = strides[d + 1] * dims[d + 1];
  }
}

// ============================================================================
// Pad implementation (aligned with old C++ code)
// ============================================================================

template <typename T>
static void pad_impl(const T *src, T *dst, int64_t ndim, const int64_t *in_dims,
                     const int64_t *out_dims, const int64_t *pad_before,
                     T constant_value) {

  // Compute total number of elements
  int64_t total_in = 1;
  for (int64_t d = 0; d < ndim; ++d) {
    total_in *= in_dims[d];
  }

  int64_t total_out = 1;
  for (int64_t d = 0; d < ndim; ++d) {
    total_out *= out_dims[d];
  }

  // First fill entire output with constant value
#pragma omp parallel for
  for (int64_t i = 0; i < total_out; ++i) {
    dst[i] = constant_value;
  }

  // Compute output strides (row-major)
  int64_t out_strides[INSIGHT_MAX_NDIM];
  compute_out_strides(ndim, out_dims, out_strides);

  // Compute input strides (row-major, since input is contiguous)
  int64_t in_strides[INSIGHT_MAX_NDIM];
  if (ndim > 0) {
    in_strides[ndim - 1] = 1;
    for (int64_t d = ndim - 2; d >= 0; --d) {
      in_strides[d] = in_strides[d + 1] * in_dims[d + 1];
    }
  }

  // Copy original data to padded region
#pragma omp parallel for
  for (int64_t linear = 0; linear < total_in; ++linear) {
    // Convert linear index to multi-dimensional indices
    int64_t indices[INSIGHT_MAX_NDIM];
    int64_t tmp = linear;
    for (int64_t d = ndim - 1; d >= 0; --d) {
      indices[d] = tmp % in_dims[d];
      tmp /= in_dims[d];
    }

    // Compute output offset
    int64_t out_offset = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      out_offset += (pad_before[d] + indices[d]) * out_strides[d];
    }

    // Compute input offset
    int64_t in_offset = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      in_offset += indices[d] * in_strides[d];
    }

    dst[out_offset] = src[in_offset];
  }
}

// ============================================================================
// C API entry point
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

C_Status pad_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("pad: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2]) {
    cpu_set_last_error("pad: pad_width is null");
    return C_FAILED;
  }

  if (!inputs[3]) {
    cpu_set_last_error("pad: constant_value is null");
    return C_FAILED;
  }

  int64_t *pad_width = (int64_t *)inputs[2];
  double constant_value = *(double *)inputs[3];

  int64_t ndim = x->ndim;

  // Input dimensions
  int64_t in_dims[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    in_dims[i] = x->dims[i];
  }

  // Output dimensions
  int64_t out_dims[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    out_dims[i] = out->dims[i];
  }

  // Pad before amounts (first half of pad_width)
  int64_t pad_before[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    pad_before[i] = pad_width[2 * i];
  }

  // Ensure output is contiguous (should be by frontend)
  if (!insight_array_is_contiguous(out)) {
    cpu_set_last_error("pad: output must be contiguous");
    return C_FAILED;
  }

  // Dispatch based on data type
  switch (x->dtype) {
  case INSIGHT_DTYPE_BOOL:
    pad_impl<bool>((const bool *)x->data, (bool *)out->data, ndim, in_dims,
                   out_dims, pad_before, (constant_value != 0));
    break;
  case INSIGHT_DTYPE_U8:
    pad_impl<uint8_t>((const uint8_t *)x->data, (uint8_t *)out->data, ndim,
                      in_dims, out_dims, pad_before, (uint8_t)constant_value);
    break;
  case INSIGHT_DTYPE_I8:
    pad_impl<int8_t>((const int8_t *)x->data, (int8_t *)out->data, ndim,
                     in_dims, out_dims, pad_before, (int8_t)constant_value);
    break;
  case INSIGHT_DTYPE_I16:
    pad_impl<int16_t>((const int16_t *)x->data, (int16_t *)out->data, ndim,
                      in_dims, out_dims, pad_before, (int16_t)constant_value);
    break;
  case INSIGHT_DTYPE_I32:
    pad_impl<int32_t>((const int32_t *)x->data, (int32_t *)out->data, ndim,
                      in_dims, out_dims, pad_before, (int32_t)constant_value);
    break;
  case INSIGHT_DTYPE_I64:
    pad_impl<int64_t>((const int64_t *)x->data, (int64_t *)out->data, ndim,
                      in_dims, out_dims, pad_before, (int64_t)constant_value);
    break;
  case INSIGHT_DTYPE_U16:
    pad_impl<uint16_t>((const uint16_t *)x->data, (uint16_t *)out->data, ndim,
                       in_dims, out_dims, pad_before, (uint16_t)constant_value);
    break;
  case INSIGHT_DTYPE_U32:
    pad_impl<uint32_t>((const uint32_t *)x->data, (uint32_t *)out->data, ndim,
                       in_dims, out_dims, pad_before, (uint32_t)constant_value);
    break;
  case INSIGHT_DTYPE_U64:
    pad_impl<uint64_t>((const uint64_t *)x->data, (uint64_t *)out->data, ndim,
                       in_dims, out_dims, pad_before, (uint64_t)constant_value);
    break;
  case INSIGHT_DTYPE_F32:
    pad_impl<float>((const float *)x->data, (float *)out->data, ndim, in_dims,
                    out_dims, pad_before, (float)constant_value);
    break;
  case INSIGHT_DTYPE_F64:
    pad_impl<double>((const double *)x->data, (double *)out->data, ndim,
                     in_dims, out_dims, pad_before, constant_value);
    break;
  case INSIGHT_DTYPE_C32: {
    std::complex<float> cval((float)constant_value, 0.0f);
    pad_impl<std::complex<float>>((const std::complex<float> *)x->data,
                                  (std::complex<float> *)out->data, ndim,
                                  in_dims, out_dims, pad_before, cval);
    break;
  }
  case INSIGHT_DTYPE_C64: {
    std::complex<double> cval(constant_value, 0.0);
    pad_impl<std::complex<double>>((const std::complex<double> *)x->data,
                                   (std::complex<double> *)out->data, ndim,
                                   in_dims, out_dims, pad_before, cval);
    break;
  }
  default:
    cpu_set_last_error("pad: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// Register for all supported types
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_BOOL, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_U8, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_I8, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_I16, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_I32, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_I64, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_U16, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_U32, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_U64, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_F32, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_F64, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_C32, pad_kernel_cpu);
REGISTER_CPU_KERNEL(pad, INSIGHT_DTYPE_C64, pad_kernel_cpu);