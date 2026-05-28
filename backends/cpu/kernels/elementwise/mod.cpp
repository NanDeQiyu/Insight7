// backends/cpu/kernels/elementwise/mod.cpp
/**
 * @file mod.cpp
 * @brief CPU kernel for modulo operation.
 *
 * Computes elementwise a % b (remainder of a divided by b) with stride support.
 * For integer types, uses C's % operator.
 * For floating point types, uses fmodf/fmod.
 * Complex numbers are not supported for modulo operation.
 *
 * @param inputs  [0] = InsightArray* dividend
 *                [1] = InsightArray* divisor
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <math.h>

/**
 * @brief Modulo for single-precision float (using fmodf).
 */
static inline float cpu_mod_float(float a, float b) { return fmodf(a, b); }

/**
 * @brief Modulo for double-precision float (using fmod).
 */
static inline double cpu_mod_double(double a, double b) { return fmod(a, b); }

#ifdef __cplusplus
extern "C" {
#endif

static void mod_loop_uint8(const uint8_t *a_data, const int64_t *a_strides,
                           const uint8_t *b_data, const int64_t *b_strides,
                           uint8_t *out_data, const int64_t *out_strides,
                           int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = a_data[off_a] % b_data[off_b];
  }
}

static void mod_loop_int8(const int8_t *a_data, const int64_t *a_strides,
                          const int8_t *b_data, const int64_t *b_strides,
                          int8_t *out_data, const int64_t *out_strides,
                          int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = a_data[off_a] % b_data[off_b];
  }
}

static void mod_loop_int16(const int16_t *a_data, const int64_t *a_strides,
                           const int16_t *b_data, const int64_t *b_strides,
                           int16_t *out_data, const int64_t *out_strides,
                           int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = a_data[off_a] % b_data[off_b];
  }
}

static void mod_loop_int32(const int32_t *a_data, const int64_t *a_strides,
                           const int32_t *b_data, const int64_t *b_strides,
                           int32_t *out_data, const int64_t *out_strides,
                           int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = a_data[off_a] % b_data[off_b];
  }
}

static void mod_loop_int64(const int64_t *a_data, const int64_t *a_strides,
                           const int64_t *b_data, const int64_t *b_strides,
                           int64_t *out_data, const int64_t *out_strides,
                           int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = a_data[off_a] % b_data[off_b];
  }
}

static void mod_loop_uint16(const uint16_t *a_data, const int64_t *a_strides,
                            const uint16_t *b_data, const int64_t *b_strides,
                            uint16_t *out_data, const int64_t *out_strides,
                            int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = a_data[off_a] % b_data[off_b];
  }
}

static void mod_loop_uint32(const uint32_t *a_data, const int64_t *a_strides,
                            const uint32_t *b_data, const int64_t *b_strides,
                            uint32_t *out_data, const int64_t *out_strides,
                            int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = a_data[off_a] % b_data[off_b];
  }
}

static void mod_loop_uint64(const uint64_t *a_data, const int64_t *a_strides,
                            const uint64_t *b_data, const int64_t *b_strides,
                            uint64_t *out_data, const int64_t *out_strides,
                            int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = a_data[off_a] % b_data[off_b];
  }
}

static void mod_loop_float(const float *a_data, const int64_t *a_strides,
                           const float *b_data, const int64_t *b_strides,
                           float *out_data, const int64_t *out_strides,
                           int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = cpu_mod_float(a_data[off_a], b_data[off_b]);
  }
}

static void mod_loop_double(const double *a_data, const int64_t *a_strides,
                            const double *b_data, const int64_t *b_strides,
                            double *out_data, const int64_t *out_strides,
                            int64_t ndim, const int64_t *dims, int64_t n) {

#pragma omp parallel for
  for (int64_t linear = 0; linear < n; ++linear) {
    int64_t off_a = cpu_offset_from_linear(linear, ndim, dims, a_strides);
    int64_t off_b = cpu_offset_from_linear(linear, ndim, dims, b_strides);
    int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);

    out_data[off_out] = cpu_mod_double(a_data[off_a], b_data[off_b]);
  }
}

C_Status mod_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *a = (InsightArray *)inputs[0];
  InsightArray *b = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a || !b || !out) {
    cpu_set_last_error("mod: null array pointer");
    return C_FAILED;
  }

  if (a->numel != out->numel || b->numel != out->numel) {
    cpu_set_last_error("mod: shape mismatch");
    return C_FAILED;
  }

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

  switch (a->dtype) {
  case INSIGHT_DTYPE_U8:
    mod_loop_uint8((const uint8_t *)a->data, a_strides,
                   (const uint8_t *)b->data, b_strides, (uint8_t *)out->data,
                   out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_I8:
    mod_loop_int8((const int8_t *)a->data, a_strides, (const int8_t *)b->data,
                  b_strides, (int8_t *)out->data, out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_I16:
    mod_loop_int16((const int16_t *)a->data, a_strides,
                   (const int16_t *)b->data, b_strides, (int16_t *)out->data,
                   out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_I32:
    mod_loop_int32((const int32_t *)a->data, a_strides,
                   (const int32_t *)b->data, b_strides, (int32_t *)out->data,
                   out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_I64:
    mod_loop_int64((const int64_t *)a->data, a_strides,
                   (const int64_t *)b->data, b_strides, (int64_t *)out->data,
                   out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_U16:
    mod_loop_uint16((const uint16_t *)a->data, a_strides,
                    (const uint16_t *)b->data, b_strides, (uint16_t *)out->data,
                    out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_U32:
    mod_loop_uint32((const uint32_t *)a->data, a_strides,
                    (const uint32_t *)b->data, b_strides, (uint32_t *)out->data,
                    out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_U64:
    mod_loop_uint64((const uint64_t *)a->data, a_strides,
                    (const uint64_t *)b->data, b_strides, (uint64_t *)out->data,
                    out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_F32:
    mod_loop_float((const float *)a->data, a_strides, (const float *)b->data,
                   b_strides, (float *)out->data, out_strides, ndim, dims, n);
    break;
  case INSIGHT_DTYPE_F64:
    mod_loop_double((const double *)a->data, a_strides, (const double *)b->data,
                    b_strides, (double *)out->data, out_strides, ndim, dims, n);
    break;
  default:
    cpu_set_last_error("mod: unsupported dtype (complex not supported)");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_U8, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_I8, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_I16, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_I32, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_I64, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_U16, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_U32, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_U64, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_F32, mod_kernel_cpu);
REGISTER_CPU_KERNEL(mod, INSIGHT_DTYPE_F64, mod_kernel_cpu);
