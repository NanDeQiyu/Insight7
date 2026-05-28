// backends/cpu/kernels/indexing/take_along_axis.cpp
/**
 * @file take_along_axis.cpp
 * @brief CPU kernel for take_along_axis operation.
 *
 * Takes values from input array using indices array along an axis.
 * Supports broadcasting.
 */

#include "common.h"
#include <complex>

// Universal template implementation
template <typename T>
static C_Status take_along_axis_impl(InsightArray *out, InsightArray *x,
                                     InsightArray *idx, int axis) {

  int64_t ndim = out->ndim;
  int64_t dims[INSIGHT_MAX_NDIM];
  int64_t out_strides[INSIGHT_MAX_NDIM];
  int64_t x_strides[INSIGHT_MAX_NDIM];

  for (int i = 0; i < ndim; ++i) {
    dims[i] = out->dims[i];
    out_strides[i] = out->strides[i];
    x_strides[i] = x->strides[i];
  }

  int64_t total_out = out->numel;
  const int64_t *idx_data = (const int64_t *)idx->data;

  T *dst = (T *)out->data;
  const T *src = (const T *)x->data;

  bool out_of_bounds = false;
  int64_t x_dim_axis = x->dims[axis];

#pragma omp parallel for
  for (int64_t linear = 0; linear < total_out; ++linear) {
    if (out_of_bounds)
      continue;

    int64_t indices[INSIGHT_MAX_NDIM];
    cpu_linear_to_indices(linear, ndim, dims, indices);

    int64_t pos = idx_data[linear];
    if (pos < 0)
      pos += x_dim_axis;
    if (pos < 0 || pos >= x_dim_axis) {
#pragma omp critical
      {
        if (!out_of_bounds) {
          cpu_set_last_error("take_along_axis: index out of bounds");
          out_of_bounds = true;
        }
      }
      continue;
    }

    // Enter offset: add x->offset to handle the view
    int64_t src_offset = x->offset;
    for (int d = 0; d < ndim; ++d) {
      if (d == axis) {
        src_offset += pos * x_strides[d];
      } else {
        src_offset += indices[d] * x_strides[d];
      }
    }

    int64_t dst_offset = out->offset;
    for (int d = 0; d < ndim; ++d) {
      dst_offset += indices[d] * out_strides[d];
    }

    dst[dst_offset] = src[src_offset];
  }

  return out_of_bounds ? C_FAILED : C_SUCCESS;
}

#define TAKE_ALONG_AXIS_CASE(DTYPE, CTYPE)                                     \
  case DTYPE:                                                                  \
    return take_along_axis_impl<CTYPE>(out, x, idx, axis);

#ifdef __cplusplus
extern "C" {
#endif

C_Status take_along_axis_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  InsightArray *idx = (InsightArray *)inputs[2];
  int axis = *(int *)inputs[3];

  if (!out || !x || !idx) {
    cpu_set_last_error("take_along_axis: null array pointer");
    return C_FAILED;
  }

  // Make sure idx is of type int64
  if (idx->dtype != INSIGHT_DTYPE_I64) {
    cpu_set_last_error("take_along_axis: indices must be int64");
    return C_FAILED;
  }

  switch (x->dtype) {
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_BOOL, bool)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_U8, uint8_t)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_I8, int8_t)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_I16, int16_t)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_I32, int32_t)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_I64, int64_t)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_U16, uint16_t)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_U32, uint32_t)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_U64, uint64_t)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_F16, uint16_t)  // float16 as uint16_t
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_BF16, uint16_t) // bfloat16 as uint16_t
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_F32, float)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_F64, double)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_C32, std::complex<float>)
    TAKE_ALONG_AXIS_CASE(INSIGHT_DTYPE_C64, std::complex<double>)
  default:
    cpu_set_last_error("take_along_axis: unsupported dtype");
    return C_FAILED;
  }
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_BOOL,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_U8,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_I8,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_I16,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_I32,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_I64,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_U16,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_U32,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_U64,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_F16,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_BF16,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_F32,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_F64,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_C32,
                    take_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(take_along_axis, INSIGHT_DTYPE_C64,
                    take_along_axis_kernel_cpu);