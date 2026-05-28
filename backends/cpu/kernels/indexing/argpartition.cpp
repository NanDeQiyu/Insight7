// backends/cpu/kernels/indexing/argpartition.cpp
/**
 * @file argpartition.cpp
 * @brief CPU kernel for argpartition operation.
 *
 * Returns indices that would partially sort the array.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (indices)
 *   inputs[1] = InsightArray* input
 *   inputs[2] = int64_t* kth
 *   inputs[3] = int* axis
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <algorithm>
#include <vector>

template <typename T>
static void argpartition_1d_impl(const T *src, int64_t *dst, int64_t n,
                                 int64_t kth) {
  std::vector<std::pair<T, int64_t>> pairs(n);
  for (int64_t i = 0; i < n; ++i) {
    pairs[i] = {src[i], i};
  }
  std::nth_element(
      pairs.begin(), pairs.begin() + kth, pairs.end(),
      [](const auto &a, const auto &b) { return a.first < b.first; });
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = pairs[i].second;
  }
}

template <typename T>
static void
argpartition_nd_impl(const T *src, int64_t *dst, int64_t ndim,
                     const int64_t *dims, const int64_t *src_strides,
                     const int64_t *dst_strides, int axis, int64_t kth) {

  int64_t axis_stride = 1;
  for (int d = axis + 1; d < ndim; ++d) {
    axis_stride *= dims[d];
  }

  int64_t axis_size = dims[axis];
  int64_t batch_size = 1;
  for (int i = 0; i < ndim; ++i) {
    if (i != axis)
      batch_size *= dims[i];
  }

  // Precompute base address for each batch
  std::vector<int64_t> batch_offsets(batch_size);
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    int64_t base_offset = 0;
    int64_t tmp = batch;
    for (int d = ndim - 1; d >= 0; --d) {
      if (d == axis)
        continue;
      int64_t coord = tmp % dims[d];
      tmp /= dims[d];
      base_offset += coord * dst_strides[d];
    }
    batch_offsets[batch] = base_offset;
  }

  for (int64_t batch = 0; batch < batch_size; ++batch) {
    std::vector<std::pair<T, int64_t>> pairs(axis_size);
    int64_t base_offset = 0;
    int64_t tmp = batch;
    for (int d = ndim - 1; d >= 0; --d) {
      if (d == axis)
        continue;
      int64_t coord = tmp % dims[d];
      tmp /= dims[d];
      base_offset += coord * src_strides[d];
    }

    for (int64_t i = 0; i < axis_size; ++i) {
      int64_t src_offset = base_offset + i * src_strides[axis];
      pairs[i] = {src[src_offset], i};
    }

    std::nth_element(
        pairs.begin(), pairs.begin() + kth, pairs.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });

    for (int64_t i = 0; i < axis_size; ++i) {
      int64_t dst_offset = batch_offsets[batch] + i * dst_strides[axis];
      dst[dst_offset] = pairs[i].second;
    }
  }
}

#define ARGPARTITION_DISPATCH(CTYPE)                                           \
  do {                                                                         \
    if (ndim == 1) {                                                           \
      argpartition_1d_impl<CTYPE>((const CTYPE *)x->data,                      \
                                  (int64_t *)out->data, n, kth);               \
    } else {                                                                   \
      argpartition_nd_impl<CTYPE>((const CTYPE *)x->data,                      \
                                  (int64_t *)out->data, ndim, dims,            \
                                  src_strides, dst_strides, axis, kth);        \
    }                                                                          \
  } while (0)

#ifdef __cplusplus
extern "C" {
#endif

C_Status argpartition_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  int64_t kth = *(int64_t *)inputs[2];
  int axis = *(int *)inputs[3];

  if (!out || !x) {
    cpu_set_last_error("argpartition: null array pointer");
    return C_FAILED;
  }

  int64_t ndim = x->ndim;
  if (axis < 0)
    axis += ndim;
  if (axis < 0 || axis >= ndim) {
    cpu_set_last_error("argpartition: axis out of range");
    return C_FAILED;
  }

  int64_t n = x->numel;
  int64_t dims[INSIGHT_MAX_NDIM];
  int64_t src_strides[INSIGHT_MAX_NDIM];
  int64_t dst_strides[INSIGHT_MAX_NDIM];

  for (int i = 0; i < ndim; ++i) {
    dims[i] = out->dims[i];
    src_strides[i] = x->strides[i];
    dst_strides[i] = out->strides[i];
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_BOOL:
    ARGPARTITION_DISPATCH(bool);
    break;
  case INSIGHT_DTYPE_U8:
    ARGPARTITION_DISPATCH(uint8_t);
    break;
  case INSIGHT_DTYPE_U16:
    ARGPARTITION_DISPATCH(uint16_t);
    break;
  case INSIGHT_DTYPE_U32:
    ARGPARTITION_DISPATCH(uint32_t);
    break;
  case INSIGHT_DTYPE_U64:
    ARGPARTITION_DISPATCH(uint64_t);
    break;
  case INSIGHT_DTYPE_I8:
    ARGPARTITION_DISPATCH(int8_t);
    break;
  case INSIGHT_DTYPE_I16:
    ARGPARTITION_DISPATCH(int16_t);
    break;
  case INSIGHT_DTYPE_I32:
    ARGPARTITION_DISPATCH(int32_t);
    break;
  case INSIGHT_DTYPE_I64:
    ARGPARTITION_DISPATCH(int64_t);
    break;
  case INSIGHT_DTYPE_F16:
    ARGPARTITION_DISPATCH(uint16_t);
    break;
  case INSIGHT_DTYPE_BF16:
    ARGPARTITION_DISPATCH(uint16_t);
    break;
  case INSIGHT_DTYPE_F32:
    ARGPARTITION_DISPATCH(float);
    break;
  case INSIGHT_DTYPE_F64:
    ARGPARTITION_DISPATCH(double);
    break;
  case INSIGHT_DTYPE_F8_E4M3:
    ARGPARTITION_DISPATCH(uint8_t);
    break;
  case INSIGHT_DTYPE_F8_E5M2:
    ARGPARTITION_DISPATCH(uint8_t);
    break;
  default:
    cpu_set_last_error("argpartition: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_BOOL, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_U8, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_U16, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_U32, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_U64, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_I8, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_I16, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_I32, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_I64, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_F16, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_BF16, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_F32, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_F64, argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_F8_E4M3,
                    argpartition_kernel_cpu);
REGISTER_CPU_KERNEL(argpartition, INSIGHT_DTYPE_F8_E5M2,
                    argpartition_kernel_cpu);