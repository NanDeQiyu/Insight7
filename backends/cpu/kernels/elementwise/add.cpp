// backends/cpu/kernels/elementwise/add.cpp
/**
 * @file add.cpp
 * @brief CPU kernel for Addition operation.
 *
 * Computes elementwise + of two arrays with stride support.
 * Supports all numeric data types including complex numbers.
 *
 * @param inputs  [0] = InsightArray* left operand
 *                [1] = InsightArray* right operand
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../common/half_utils.h"
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status add_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *a = (InsightArray *)inputs[0];
  InsightArray *b = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a || !b || !out) {
    cpu_set_last_error("add: null array pointer");
    return C_FAILED;
  }

  // Check that shapes are compatible (numel must match)
  if (a->numel != out->numel || b->numel != out->numel) {
    cpu_set_last_error("add: shape mismatch");
    return C_FAILED;
  }

  switch (a->dtype) {
  case INSIGHT_DTYPE_BOOL:
    ELEMENTWISE_KERNEL_LOOP(bool, +);
    break;
  case INSIGHT_DTYPE_U8:
    ELEMENTWISE_KERNEL_LOOP(uint8_t, +);
    break;
  case INSIGHT_DTYPE_I8:
    ELEMENTWISE_KERNEL_LOOP(int8_t, +);
    break;
  case INSIGHT_DTYPE_I16:
    ELEMENTWISE_KERNEL_LOOP(int16_t, +);
    break;
  case INSIGHT_DTYPE_I32:
    ELEMENTWISE_KERNEL_LOOP(int32_t, +);
    break;
  case INSIGHT_DTYPE_I64:
    ELEMENTWISE_KERNEL_LOOP(int64_t, +);
    break;
  case INSIGHT_DTYPE_U16:
    ELEMENTWISE_KERNEL_LOOP(uint16_t, +);
    break;
  case INSIGHT_DTYPE_U32:
    ELEMENTWISE_KERNEL_LOOP(uint32_t, +);
    break;
  case INSIGHT_DTYPE_U64:
    ELEMENTWISE_KERNEL_LOOP(uint64_t, +);
    break;
  case INSIGHT_DTYPE_F32:
    ELEMENTWISE_KERNEL_LOOP(float, +);
    break;
  case INSIGHT_DTYPE_F64:
    ELEMENTWISE_KERNEL_LOOP(double, +);
    break;
  case INSIGHT_DTYPE_C32:
    ELEMENTWISE_KERNEL_LOOP(std::complex<float>, +);
    break;
  case INSIGHT_DTYPE_C64:
    ELEMENTWISE_KERNEL_LOOP(std::complex<double>, +);
    break;
  case INSIGHT_DTYPE_F16: {
    const uint16_t *a_data = (const uint16_t *)a->data;
    const uint16_t *b_data = (const uint16_t *)b->data;
    uint16_t *out_data = (uint16_t *)out->data;
    int64_t ndim = out->ndim;
    int64_t dims[INSIGHT_MAX_NDIM];
    int64_t a_strides[INSIGHT_MAX_NDIM];
    int64_t b_strides[INSIGHT_MAX_NDIM];
    int64_t out_strides[INSIGHT_MAX_NDIM];
    for (int i = 0; i < ndim; ++i) {
      dims[i] = out->dims[i];
      a_strides[i] = a->strides[i];
      b_strides[i] = b->strides[i];
      out_strides[i] = out->strides[i];
    }
    int64_t n = out->numel;
#pragma omp parallel for
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_a =
          a->offset + cpu_offset_from_linear(linear, ndim, dims, a_strides);
      int64_t off_b =
          b->offset + cpu_offset_from_linear(linear, ndim, dims, b_strides);
      int64_t off_out =
          out->offset + cpu_offset_from_linear(linear, ndim, dims, out_strides);
      float va = insight::f16_to_f32(a_data[off_a]);
      float vb = insight::f16_to_f32(b_data[off_b]);
      out_data[off_out] = insight::f32_to_f16(va + vb);
    }
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    const uint16_t *a_data = (const uint16_t *)a->data;
    const uint16_t *b_data = (const uint16_t *)b->data;
    uint16_t *out_data = (uint16_t *)out->data;
    int64_t ndim = out->ndim;
    int64_t dims[INSIGHT_MAX_NDIM];
    int64_t a_strides[INSIGHT_MAX_NDIM];
    int64_t b_strides[INSIGHT_MAX_NDIM];
    int64_t out_strides[INSIGHT_MAX_NDIM];
    for (int i = 0; i < ndim; ++i) {
      dims[i] = out->dims[i];
      a_strides[i] = a->strides[i];
      b_strides[i] = b->strides[i];
      out_strides[i] = out->strides[i];
    }
    int64_t n = out->numel;
#pragma omp parallel for
    for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_a =
          a->offset + cpu_offset_from_linear(linear, ndim, dims, a_strides);
      int64_t off_b =
          b->offset + cpu_offset_from_linear(linear, ndim, dims, b_strides);
      int64_t off_out =
          out->offset + cpu_offset_from_linear(linear, ndim, dims, out_strides);
      float va = insight::bf16_to_f32(a_data[off_a]);
      float vb = insight::bf16_to_f32(b_data[off_b]);
      out_data[off_out] = insight::f32_to_bf16(va + vb);
    }
    break;
  }
  default:
    cpu_set_last_error("add: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// Register for all supported types
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_BOOL, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_U8, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_I8, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_I16, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_I32, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_I64, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_U16, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_U32, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_U64, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_F16, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_BF16, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_F32, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_F64, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_C32, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_C64, add_kernel_cpu);
