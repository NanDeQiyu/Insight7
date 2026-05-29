// backends/cuda/kernels/indexing/topk.cu
/**
 * @file topk.cu
 * @brief CUDA kernel for topk operation.
 *
 * Returns the k largest/smallest elements and their indices.
 * Uses CPU fallback.
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* values output
 *   inputs[1] = InsightArray* indices output
 *   inputs[2] = InsightArray* input (already transposed so axis is last)
 *   inputs[3] = int64_t* k
 *   inputs[4] = bool* largest
 *   inputs[5] = bool* sorted
 */

#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <utility>

extern "C" {

C_Status topk_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out_values = static_cast<InsightArray *>(outputs[0]);
  InsightArray *out_indices = static_cast<InsightArray *>(outputs[1]);
  InsightArray *x = static_cast<InsightArray *>(inputs[2]);

  if (!out_values || !out_indices || !x) {
    gpu_set_last_error("topk: null array pointer");
    return C_FAILED;
  }

  int k = *static_cast<int *>(inputs[3]);
  bool largest = *static_cast<bool *>(inputs[4]);
  bool sorted_flag = *static_cast<bool *>(inputs[5]);

  int64_t ndim = x->ndim;
  int64_t n = x->numel;
  int64_t axis_size = x->dims[ndim - 1]; // axis is always last after transpose
  int64_t outer_size = n / axis_size;

  // Copy input to CPU
  float *host_data = new float[n];
  cudaMemcpy(host_data, x->data, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Compute topk on CPU
  int64_t out_n = outer_size * k;
  float *values_data = new float[out_n];
  int64_t *indices_data = new int64_t[out_n];

  for (int64_t outer = 0; outer < outer_size; ++outer) {
    int64_t base = outer * axis_size;
    int64_t out_base = outer * k;

    // Create index-value pairs
    std::pair<float, int64_t> *pairs = new std::pair<float, int64_t>[axis_size];
    for (int64_t i = 0; i < axis_size; ++i) {
      pairs[i] = {host_data[base + i], i};
    }

    // Partial sort to get top k
    if (largest) {
      std::partial_sort(
          pairs, pairs + k, pairs + axis_size,
          [](const auto &a, const auto &b) { return a.first > b.first; });
    } else {
      std::partial_sort(
          pairs, pairs + k, pairs + axis_size,
          [](const auto &a, const auto &b) { return a.first < b.first; });
    }

    // Write results
    for (int64_t i = 0; i < k; ++i) {
      values_data[out_base + i] = pairs[i].first;
      indices_data[out_base + i] = pairs[i].second;
    }
    delete[] pairs;
  }

  // Copy results back to GPU
  cudaMemcpy(out_values->data, values_data, out_n * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(out_indices->data, indices_data, out_n * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  delete[] host_data;
  delete[] values_data;
  delete[] indices_data;

  return C_SUCCESS;
}

} // extern "C"

REGISTER_GPU_KERNEL(topk, INSIGHT_DTYPE_F32, topk_kernel_gpu);
REGISTER_GPU_KERNEL(topk, INSIGHT_DTYPE_F64, topk_kernel_gpu);
