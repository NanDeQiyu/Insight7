// backends/cuda/kernels/indexing/masked_select.cu
/**
 * @file masked_select.cu
 * @brief CUDA kernel for masked_select operation.
 *
 * Returns elements selected by a boolean mask.
 * Uses CPU fallback.
 */

#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

extern "C" {

C_Status masked_select_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);
  InsightArray *mask = static_cast<InsightArray *>(inputs[2]);

  if (!out || !x || !mask) {
    gpu_set_last_error("masked_select: null array pointer");
    return C_FAILED;
  }

  int64_t n = x->numel;

  // Copy inputs to CPU
  float *x_data = new float[n];
  bool *mask_data = new bool[n];
  cudaMemcpy(x_data, x->data, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(mask_data, mask->data, n * sizeof(bool), cudaMemcpyDeviceToHost);

  // Select elements where mask is true
  int64_t count = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (mask_data[i]) {
      ++count;
    }
  }

  float *result = new float[count];
  int64_t idx = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (mask_data[i]) {
      result[idx++] = x_data[i];
    }
  }

  // Copy result back to GPU
  if (count > 0) {
    cudaMemcpy(out->data, result, count * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  delete[] result;
  delete[] x_data;
  delete[] mask_data;

  return C_SUCCESS;
}

} // extern "C"

REGISTER_GPU_KERNEL(masked_select, INSIGHT_DTYPE_F32, masked_select_kernel_gpu);
