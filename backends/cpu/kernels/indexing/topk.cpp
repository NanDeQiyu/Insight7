// backends/cpu/kernels/indexing/topk.cpp
/**
 * @file topk.cpp
 * @brief CPU kernel for topk operation.
 *
 * Returns top k values and their indices.
 *
 * Input layout:
 *   inputs[0] = InsightArray* values output
 *   inputs[1] = InsightArray* indices output
 *   inputs[2] = InsightArray* input
 *   inputs[3] = int64_t* k
 *   inputs[4] = bool* largest
 *   inputs[5] = bool* sorted
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* values, [1] = InsightArray* indices
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../common/half_utils.h"
#include "common.h"
#include <algorithm>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

C_Status topk_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *values = (InsightArray *)outputs[0];
  InsightArray *indices = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];
  int64_t k = *(int64_t *)inputs[3];
  bool largest = *(bool *)inputs[4];
  bool sorted = *(bool *)inputs[5];

  if (!values || !indices || !x) {
    cpu_set_last_error("topk: null array pointer");
    return C_FAILED;
  }

  int64_t last_dim = x->dims[x->ndim - 1];
  int64_t batch_size = x->numel / last_dim;
  void *vals_data = values->data;
  int64_t *idx_data = (int64_t *)indices->data;

  if (k > last_dim)
    k = last_dim;

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32: {
    const float *src = (const float *)x->data;
    float *dst_vals = (float *)vals_data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<std::pair<float, int64_t>> pairs(last_dim);
      for (int64_t i = 0; i < last_dim; ++i) {
        pairs[i] = {src[batch * last_dim + i], i};
      }
      if (largest) {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
      } else {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
      }
      for (int64_t i = 0; i < k; ++i) {
        dst_vals[batch * k + i] = pairs[i].first;
        idx_data[batch * k + i] = pairs[i].second;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_F64: {
    const double *src = (const double *)x->data;
    double *dst_vals = (double *)vals_data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<std::pair<double, int64_t>> pairs(last_dim);
      for (int64_t i = 0; i < last_dim; ++i) {
        pairs[i] = {src[batch * last_dim + i], i};
      }
      if (largest) {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
      } else {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
      }
      for (int64_t i = 0; i < k; ++i) {
        dst_vals[batch * k + i] = pairs[i].first;
        idx_data[batch * k + i] = pairs[i].second;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_I32: {
    const int32_t *src = (const int32_t *)x->data;
    int32_t *dst_vals = (int32_t *)vals_data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<std::pair<int32_t, int64_t>> pairs(last_dim);
      for (int64_t i = 0; i < last_dim; ++i) {
        pairs[i] = {src[batch * last_dim + i], i};
      }
      if (largest) {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
      } else {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
      }
      for (int64_t i = 0; i < k; ++i) {
        dst_vals[batch * k + i] = pairs[i].first;
        idx_data[batch * k + i] = pairs[i].second;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_I64: {
    const int64_t *src = (const int64_t *)x->data;
    int64_t *dst_vals = (int64_t *)vals_data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<std::pair<int64_t, int64_t>> pairs(last_dim);
      for (int64_t i = 0; i < last_dim; ++i) {
        pairs[i] = {src[batch * last_dim + i], i};
      }
      if (largest) {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
      } else {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
      }
      for (int64_t i = 0; i < k; ++i) {
        dst_vals[batch * k + i] = pairs[i].first;
        idx_data[batch * k + i] = pairs[i].second;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_F16: {
    const uint16_t *src = (const uint16_t *)x->data;
    uint16_t *dst_vals = (uint16_t *)vals_data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<std::pair<float, int64_t>> pairs(last_dim);
      for (int64_t i = 0; i < last_dim; ++i) {
        pairs[i] = {insight::f16_to_f32(src[batch * last_dim + i]), i};
      }
      if (largest) {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
      } else {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
      }
      for (int64_t i = 0; i < k; ++i) {
        dst_vals[batch * k + i] = insight::f32_to_f16(pairs[i].first);
        idx_data[batch * k + i] = pairs[i].second;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_BF16: {
    const uint16_t *src = (const uint16_t *)x->data;
    uint16_t *dst_vals = (uint16_t *)vals_data;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      std::vector<std::pair<float, int64_t>> pairs(last_dim);
      for (int64_t i = 0; i < last_dim; ++i) {
        pairs[i] = {insight::bf16_to_f32(src[batch * last_dim + i]), i};
      }
      if (largest) {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
      } else {
        std::partial_sort(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
      }
      for (int64_t i = 0; i < k; ++i) {
        dst_vals[batch * k + i] = insight::f32_to_bf16(pairs[i].first);
        idx_data[batch * k + i] = pairs[i].second;
      }
    }
    break;
  }
  default:
    cpu_set_last_error("topk: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(topk, INSIGHT_DTYPE_F32, topk_kernel_cpu);
REGISTER_CPU_KERNEL(topk, INSIGHT_DTYPE_F64, topk_kernel_cpu);
REGISTER_CPU_KERNEL(topk, INSIGHT_DTYPE_I32, topk_kernel_cpu);
REGISTER_CPU_KERNEL(topk, INSIGHT_DTYPE_I64, topk_kernel_cpu);
REGISTER_CPU_KERNEL(topk, INSIGHT_DTYPE_F16, topk_kernel_cpu);
REGISTER_CPU_KERNEL(topk, INSIGHT_DTYPE_BF16, topk_kernel_cpu);
