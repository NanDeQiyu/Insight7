// backends/cpu/kernels/indexing/argsort.cpp
/**
 * @file argsort.cpp
 * @brief CPU kernel for argsort operation.
 *
 * Returns indices that would sort the array.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (indices, int64)
 *   inputs[1] = InsightArray* input
 *   inputs[2] = bool* descending
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../common/half_utils.h"
#include "common.h"
#include <algorithm>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

C_Status argsort_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  bool descending = *(bool *)inputs[2];

  if (!out || !x) {
    cpu_set_last_error("argsort: null array pointer");
    return C_FAILED;
  }

  int64_t last_dim = x->dims[x->ndim - 1];
  int64_t batch_size = x->numel / last_dim;
  int64_t *out_data = (int64_t *)out->data;

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32: {
    const float *src = (const float *)x->data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<std::pair<float, int64_t>> pairs(last_dim);
      for (int64_t i = 0; i < last_dim; ++i) {
        pairs[i] = {src[batch * last_dim + i], i};
      }
      if (descending) {
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
          return a.first > b.first;
        });
      } else {
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
          return a.first < b.first;
        });
      }
      for (int64_t i = 0; i < last_dim; ++i) {
        out_data[batch * last_dim + i] = pairs[i].second;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_F64: {
    const double *src = (const double *)x->data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<std::pair<double, int64_t>> pairs(last_dim);
      for (int64_t i = 0; i < last_dim; ++i) {
        pairs[i] = {src[batch * last_dim + i], i};
      }
      if (descending) {
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
          return a.first > b.first;
        });
      } else {
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
          return a.first < b.first;
        });
      }
      for (int64_t i = 0; i < last_dim; ++i) {
        out_data[batch * last_dim + i] = pairs[i].second;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_I32: {
    const int32_t *src = (const int32_t *)x->data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<std::pair<int32_t, int64_t>> pairs(last_dim);
      for (int64_t i = 0; i < last_dim; ++i) {
        pairs[i] = {src[batch * last_dim + i], i};
      }
      if (descending) {
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
          return a.first > b.first;
        });
      } else {
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
          return a.first < b.first;
        });
      }
      for (int64_t i = 0; i < last_dim; ++i) {
        out_data[batch * last_dim + i] = pairs[i].second;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_I64: {
    const int64_t *src = (const int64_t *)x->data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<std::pair<int64_t, int64_t>> pairs(last_dim);
      for (int64_t i = 0; i < last_dim; ++i) {
        pairs[i] = {src[batch * last_dim + i], i};
      }
      if (descending) {
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
          return a.first > b.first;
        });
      } else {
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
          return a.first < b.first;
        });
      }
      for (int64_t i = 0; i < last_dim; ++i) {
        out_data[batch * last_dim + i] = pairs[i].second;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_F16: {
    const uint16_t *src = (const uint16_t *)x->data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<int64_t> indices(last_dim);
      for (int64_t i = 0; i < last_dim; ++i)
        indices[i] = i;
      if (descending) {
        std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
          return insight::f16_to_f32(src[batch * last_dim + a]) >
                 insight::f16_to_f32(src[batch * last_dim + b]);
        });
      } else {
        std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
          return insight::f16_to_f32(src[batch * last_dim + a]) <
                 insight::f16_to_f32(src[batch * last_dim + b]);
        });
      }
      for (int64_t i = 0; i < last_dim; ++i) {
        out_data[batch * last_dim + i] = indices[i];
      }
    }
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    const uint16_t *src = (const uint16_t *)x->data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<int64_t> indices(last_dim);
      for (int64_t i = 0; i < last_dim; ++i)
        indices[i] = i;
      if (descending) {
        std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
          return insight::bf16_to_f32(src[batch * last_dim + a]) >
                 insight::bf16_to_f32(src[batch * last_dim + b]);
        });
      } else {
        std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
          return insight::bf16_to_f32(src[batch * last_dim + a]) <
                 insight::bf16_to_f32(src[batch * last_dim + b]);
        });
      }
      for (int64_t i = 0; i < last_dim; ++i) {
        out_data[batch * last_dim + i] = indices[i];
      }
    }
    break;
  }
  default:
    cpu_set_last_error("argsort: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(argsort, INSIGHT_DTYPE_F32, argsort_kernel_cpu);
REGISTER_CPU_KERNEL(argsort, INSIGHT_DTYPE_F64, argsort_kernel_cpu);
REGISTER_CPU_KERNEL(argsort, INSIGHT_DTYPE_I32, argsort_kernel_cpu);
REGISTER_CPU_KERNEL(argsort, INSIGHT_DTYPE_I64, argsort_kernel_cpu);
REGISTER_CPU_KERNEL(argsort, INSIGHT_DTYPE_F16, argsort_kernel_cpu);
REGISTER_CPU_KERNEL(argsort, INSIGHT_DTYPE_BF16, argsort_kernel_cpu);
