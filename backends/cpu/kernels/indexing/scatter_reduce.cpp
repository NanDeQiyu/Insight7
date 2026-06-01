// backends/cpu/kernels/indexing/scatter_reduce.cpp
/**
 * @file scatter_reduce.cpp
 * @brief CPU kernel for scatter_reduce operation.
 *
 * Scatters values into an array with reduction.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (will be modified)
 *   inputs[1] = InsightArray* indices
 *   inputs[2] = InsightArray* src
 *   inputs[3] = int* dim
 *   inputs[4] = char* reduce (string: "replace", "add", "mul", "max", "min")
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../common/half_utils.h"
#include "common.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SCATTER_REDUCE_IMPL(CTYPE, OP)                                         \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src_data = (const CTYPE *)src->data;                          \
    int64_t total = idx->numel;                                                \
    int64_t src_size = src->numel;                                             \
    for (int64_t linear = 0; linear < total; ++linear) {                       \
      int64_t indices[INSIGHT_MAX_NDIM];                                       \
      int64_t tmp = linear;                                                    \
      for (int d = ndim - 1; d >= 0; --d) {                                    \
        indices[d] = tmp % dims[d];                                            \
        tmp /= dims[d];                                                        \
      }                                                                        \
      int64_t pos = idx_data[linear];                                          \
      if (pos < 0)                                                             \
        pos += out->dims[dim];                                                 \
      if (pos < 0 || pos >= out->dims[dim]) {                                  \
        cpu_set_last_error("scatter_reduce: index out of bounds");             \
        return C_FAILED;                                                       \
      }                                                                        \
      int64_t out_offset = 0;                                                  \
      for (int d = 0; d < ndim; ++d) {                                         \
        if (d == dim) {                                                        \
          out_offset += pos * out_strides[d];                                  \
        } else {                                                               \
          out_offset += indices[d] * out_strides[d];                           \
        }                                                                      \
      }                                                                        \
      dst[out_offset] OP src_data[linear % src_size];                          \
    }                                                                          \
  } while (0)

#define SCATTER_REDUCE_IMPL_REPLACE(CTYPE)                                     \
  do {                                                                         \
    CTYPE *dst = (CTYPE *)out->data;                                           \
    const CTYPE *src_data = (const CTYPE *)src->data;                          \
    int64_t total = idx->numel;                                                \
    int64_t src_size = src->numel;                                             \
    for (int64_t linear = 0; linear < total; ++linear) {                       \
      int64_t indices[INSIGHT_MAX_NDIM];                                       \
      int64_t tmp = linear;                                                    \
      for (int d = ndim - 1; d >= 0; --d) {                                    \
        indices[d] = tmp % dims[d];                                            \
        tmp /= dims[d];                                                        \
      }                                                                        \
      int64_t pos = idx_data[linear];                                          \
      if (pos < 0)                                                             \
        pos += out->dims[dim];                                                 \
      if (pos < 0 || pos >= out->dims[dim]) {                                  \
        cpu_set_last_error("scatter_reduce: index out of bounds");             \
        return C_FAILED;                                                       \
      }                                                                        \
      int64_t out_offset = 0;                                                  \
      for (int d = 0; d < ndim; ++d) {                                         \
        if (d == dim) {                                                        \
          out_offset += pos * out_strides[d];                                  \
        } else {                                                               \
          out_offset += indices[d] * out_strides[d];                           \
        }                                                                      \
      }                                                                        \
      dst[out_offset] = src_data[linear % src_size];                           \
    }                                                                          \
  } while (0)

C_Status scatter_reduce_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *idx = (InsightArray *)inputs[1];
  InsightArray *src = (InsightArray *)inputs[2];
  int dim = *(int *)inputs[3];
  char *reduce = (char *)inputs[4];

  if (!out || !idx || !src) {
    cpu_set_last_error("scatter_reduce: null array pointer");
    return C_FAILED;
  }

  int64_t ndim = out->ndim;
  int64_t dims[INSIGHT_MAX_NDIM];
  int64_t out_strides[INSIGHT_MAX_NDIM];

  for (int i = 0; i < ndim; ++i) {
    dims[i] = out->dims[i];
    out_strides[i] = out->strides[i];
  }

  int64_t *idx_data = (int64_t *)idx->data;

  // Determine reduction type
  bool replace = (strcmp(reduce, "replace") == 0);
  bool add = (strcmp(reduce, "add") == 0);
  bool mul = (strcmp(reduce, "mul") == 0);
  bool max_op = (strcmp(reduce, "max") == 0);
  bool min_op = (strcmp(reduce, "min") == 0);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    if (replace) {
      SCATTER_REDUCE_IMPL_REPLACE(float);
    } else if (add) {
      SCATTER_REDUCE_IMPL(float, +=);
    } else if (mul) {
      SCATTER_REDUCE_IMPL(float, *=);
    } else if (max_op) {
      float *dst = (float *)out->data;
      const float *src_data = (const float *)src->data;
      int64_t total = idx->numel;
      int64_t src_size = src->numel;
      for (int64_t linear = 0; linear < total; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t pos = idx_data[linear];
        if (pos < 0)
          pos += out->dims[dim];
        if (pos < 0 || pos >= out->dims[dim]) {
          cpu_set_last_error("scatter_reduce: index out of bounds");
          return C_FAILED;
        }
        int64_t out_offset = 0;
        for (int d = 0; d < ndim; ++d) {
          if (d == dim) {
            out_offset += pos * out_strides[d];
          } else {
            out_offset += indices[d] * out_strides[d];
          }
        }
        float val = src_data[linear % src_size];
        if (val > dst[out_offset])
          dst[out_offset] = val;
      }
    } else if (min_op) {
      float *dst = (float *)out->data;
      const float *src_data = (const float *)src->data;
      int64_t total = idx->numel;
      int64_t src_size = src->numel;
      for (int64_t linear = 0; linear < total; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t pos = idx_data[linear];
        if (pos < 0)
          pos += out->dims[dim];
        if (pos < 0 || pos >= out->dims[dim]) {
          cpu_set_last_error("scatter_reduce: index out of bounds");
          return C_FAILED;
        }
        int64_t out_offset = 0;
        for (int d = 0; d < ndim; ++d) {
          if (d == dim) {
            out_offset += pos * out_strides[d];
          } else {
            out_offset += indices[d] * out_strides[d];
          }
        }
        float val = src_data[linear % src_size];
        if (val < dst[out_offset])
          dst[out_offset] = val;
      }
    }
    break;
  case INSIGHT_DTYPE_F64:
    if (replace) {
      SCATTER_REDUCE_IMPL_REPLACE(double);
    } else if (add) {
      SCATTER_REDUCE_IMPL(double, +=);
    } else if (mul) {
      SCATTER_REDUCE_IMPL(double, *=);
    } else if (max_op) {
      double *dst = (double *)out->data;
      const double *src_data = (const double *)src->data;
      int64_t total = idx->numel;
      int64_t src_size = src->numel;
      for (int64_t linear = 0; linear < total; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t pos = idx_data[linear];
        if (pos < 0)
          pos += out->dims[dim];
        if (pos < 0 || pos >= out->dims[dim]) {
          cpu_set_last_error("scatter_reduce: index out of bounds");
          return C_FAILED;
        }
        int64_t out_offset = 0;
        for (int d = 0; d < ndim; ++d) {
          if (d == dim) {
            out_offset += pos * out_strides[d];
          } else {
            out_offset += indices[d] * out_strides[d];
          }
        }
        double val = src_data[linear % src_size];
        if (val > dst[out_offset])
          dst[out_offset] = val;
      }
    } else if (min_op) {
      double *dst = (double *)out->data;
      const double *src_data = (const double *)src->data;
      int64_t total = idx->numel;
      int64_t src_size = src->numel;
      for (int64_t linear = 0; linear < total; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t pos = idx_data[linear];
        if (pos < 0)
          pos += out->dims[dim];
        if (pos < 0 || pos >= out->dims[dim]) {
          cpu_set_last_error("scatter_reduce: index out of bounds");
          return C_FAILED;
        }
        int64_t out_offset = 0;
        for (int d = 0; d < ndim; ++d) {
          if (d == dim) {
            out_offset += pos * out_strides[d];
          } else {
            out_offset += indices[d] * out_strides[d];
          }
        }
        double val = src_data[linear % src_size];
        if (val < dst[out_offset])
          dst[out_offset] = val;
      }
    }
    break;
  case INSIGHT_DTYPE_I32:
    if (replace) {
      SCATTER_REDUCE_IMPL_REPLACE(int32_t);
    } else if (add) {
      SCATTER_REDUCE_IMPL(int32_t, +=);
    } else if (mul) {
      SCATTER_REDUCE_IMPL(int32_t, *=);
    } else if (max_op) {
      int32_t *dst = (int32_t *)out->data;
      const int32_t *src_data = (const int32_t *)src->data;
      int64_t total = idx->numel;
      int64_t src_size = src->numel;
      for (int64_t linear = 0; linear < total; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t pos = idx_data[linear];
        if (pos < 0)
          pos += out->dims[dim];
        if (pos < 0 || pos >= out->dims[dim]) {
          cpu_set_last_error("scatter_reduce: index out of bounds");
          return C_FAILED;
        }
        int64_t out_offset = 0;
        for (int d = 0; d < ndim; ++d) {
          if (d == dim) {
            out_offset += pos * out_strides[d];
          } else {
            out_offset += indices[d] * out_strides[d];
          }
        }
        int32_t val = src_data[linear % src_size];
        if (val > dst[out_offset])
          dst[out_offset] = val;
      }
    } else if (min_op) {
      int32_t *dst = (int32_t *)out->data;
      const int32_t *src_data = (const int32_t *)src->data;
      int64_t total = idx->numel;
      int64_t src_size = src->numel;
      for (int64_t linear = 0; linear < total; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t pos = idx_data[linear];
        if (pos < 0)
          pos += out->dims[dim];
        if (pos < 0 || pos >= out->dims[dim]) {
          cpu_set_last_error("scatter_reduce: index out of bounds");
          return C_FAILED;
        }
        int64_t out_offset = 0;
        for (int d = 0; d < ndim; ++d) {
          if (d == dim) {
            out_offset += pos * out_strides[d];
          } else {
            out_offset += indices[d] * out_strides[d];
          }
        }
        int32_t val = src_data[linear % src_size];
        if (val < dst[out_offset])
          dst[out_offset] = val;
      }
    }
    break;
  case INSIGHT_DTYPE_I64:
    if (replace) {
      SCATTER_REDUCE_IMPL_REPLACE(int64_t);
    } else if (add) {
      SCATTER_REDUCE_IMPL(int64_t, +=);
    } else if (mul) {
      SCATTER_REDUCE_IMPL(int64_t, *=);
    } else if (max_op) {
      int64_t *dst = (int64_t *)out->data;
      const int64_t *src_data = (const int64_t *)src->data;
      int64_t total = idx->numel;
      int64_t src_size = src->numel;
      for (int64_t linear = 0; linear < total; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t pos = idx_data[linear];
        if (pos < 0)
          pos += out->dims[dim];
        if (pos < 0 || pos >= out->dims[dim]) {
          cpu_set_last_error("scatter_reduce: index out of bounds");
          return C_FAILED;
        }
        int64_t out_offset = 0;
        for (int d = 0; d < ndim; ++d) {
          if (d == dim) {
            out_offset += pos * out_strides[d];
          } else {
            out_offset += indices[d] * out_strides[d];
          }
        }
        int64_t val = src_data[linear % src_size];
        if (val > dst[out_offset])
          dst[out_offset] = val;
      }
    } else if (min_op) {
      int64_t *dst = (int64_t *)out->data;
      const int64_t *src_data = (const int64_t *)src->data;
      int64_t total = idx->numel;
      int64_t src_size = src->numel;
      for (int64_t linear = 0; linear < total; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        int64_t pos = idx_data[linear];
        if (pos < 0)
          pos += out->dims[dim];
        if (pos < 0 || pos >= out->dims[dim]) {
          cpu_set_last_error("scatter_reduce: index out of bounds");
          return C_FAILED;
        }
        int64_t out_offset = 0;
        for (int d = 0; d < ndim; ++d) {
          if (d == dim) {
            out_offset += pos * out_strides[d];
          } else {
            out_offset += indices[d] * out_strides[d];
          }
        }
        int64_t val = src_data[linear % src_size];
        if (val < dst[out_offset])
          dst[out_offset] = val;
      }
    }
    break;
  case INSIGHT_DTYPE_F16: {
    uint16_t *dst = (uint16_t *)out->data;
    const uint16_t *src_data = (const uint16_t *)src->data;
    int64_t total = idx->numel;
    int64_t src_size = src->numel;
    for (int64_t linear = 0; linear < total; ++linear) {
      int64_t indices[INSIGHT_MAX_NDIM];
      int64_t tmp = linear;
      for (int d = ndim - 1; d >= 0; --d) {
        indices[d] = tmp % dims[d];
        tmp /= dims[d];
      }
      int64_t pos = idx_data[linear];
      if (pos < 0)
        pos += out->dims[dim];
      int64_t out_offset = 0;
      for (int d = 0; d < ndim; ++d) {
        if (d == dim)
          out_offset += pos * out_strides[d];
        else
          out_offset += indices[d] * out_strides[d];
      }
      uint16_t sval = src_data[linear % src_size];
      if (replace) {
        dst[out_offset] = sval;
      } else if (add) {
        dst[out_offset] = insight::f32_to_f16(
            insight::f16_to_f32(dst[out_offset]) + insight::f16_to_f32(sval));
      } else if (mul) {
        dst[out_offset] = insight::f32_to_f16(
            insight::f16_to_f32(dst[out_offset]) * insight::f16_to_f32(sval));
      } else if (max_op) {
        float dv = insight::f16_to_f32(dst[out_offset]);
        float sv = insight::f16_to_f32(sval);
        if (sv > dv)
          dst[out_offset] = sval;
      } else if (min_op) {
        float dv = insight::f16_to_f32(dst[out_offset]);
        float sv = insight::f16_to_f32(sval);
        if (sv < dv)
          dst[out_offset] = sval;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    uint16_t *dst = (uint16_t *)out->data;
    const uint16_t *src_data = (const uint16_t *)src->data;
    int64_t total = idx->numel;
    int64_t src_size = src->numel;
    for (int64_t linear = 0; linear < total; ++linear) {
      int64_t indices[INSIGHT_MAX_NDIM];
      int64_t tmp = linear;
      for (int d = ndim - 1; d >= 0; --d) {
        indices[d] = tmp % dims[d];
        tmp /= dims[d];
      }
      int64_t pos = idx_data[linear];
      if (pos < 0)
        pos += out->dims[dim];
      int64_t out_offset = 0;
      for (int d = 0; d < ndim; ++d) {
        if (d == dim)
          out_offset += pos * out_strides[d];
        else
          out_offset += indices[d] * out_strides[d];
      }
      uint16_t sval = src_data[linear % src_size];
      if (replace) {
        dst[out_offset] = sval;
      } else if (add) {
        dst[out_offset] = insight::f32_to_bf16(
            insight::bf16_to_f32(dst[out_offset]) + insight::bf16_to_f32(sval));
      } else if (mul) {
        dst[out_offset] = insight::f32_to_bf16(
            insight::bf16_to_f32(dst[out_offset]) * insight::bf16_to_f32(sval));
      } else if (max_op) {
        float dv = insight::bf16_to_f32(dst[out_offset]);
        float sv = insight::bf16_to_f32(sval);
        if (sv > dv)
          dst[out_offset] = sval;
      } else if (min_op) {
        float dv = insight::bf16_to_f32(dst[out_offset]);
        float sv = insight::bf16_to_f32(sval);
        if (sv < dv)
          dst[out_offset] = sval;
      }
    }
    break;
  }
  default:
    cpu_set_last_error("scatter_reduce: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(scatter_reduce, INSIGHT_DTYPE_F32,
                    scatter_reduce_kernel_cpu);
REGISTER_CPU_KERNEL(scatter_reduce, INSIGHT_DTYPE_F64,
                    scatter_reduce_kernel_cpu);
REGISTER_CPU_KERNEL(scatter_reduce, INSIGHT_DTYPE_I32,
                    scatter_reduce_kernel_cpu);
REGISTER_CPU_KERNEL(scatter_reduce, INSIGHT_DTYPE_I64,
                    scatter_reduce_kernel_cpu);
REGISTER_CPU_KERNEL(scatter_reduce, INSIGHT_DTYPE_F16,
                    scatter_reduce_kernel_cpu);
REGISTER_CPU_KERNEL(scatter_reduce, INSIGHT_DTYPE_BF16,
                    scatter_reduce_kernel_cpu);
