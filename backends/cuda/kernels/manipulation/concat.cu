// backends/cuda/kernels/manipulation/concat.cu
/**
 * @file concat.cu
 * @brief CUDA kernel for concatenation operation.
 *
 * Concatenates multiple arrays along an axis.
 * Uses CPU fallback for complex logic.
 */

#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

extern "C" {

C_Status concat_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("concat: output is null");
    return C_FAILED;
  }

  if (!inputs[1]) {
    gpu_set_last_error("concat: num_inputs is null");
    return C_FAILED;
  }

  int num_inputs = *static_cast<int *>(inputs[1]);

  if (num_inputs < 1) {
    gpu_set_last_error("concat: no input arrays");
    return C_FAILED;
  }

  // Extract input arrays
  std::vector<InsightArray *> in_arrays;
  in_arrays.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    if (!inputs[2 + i]) {
      gpu_set_last_error("concat: null input array");
      return C_FAILED;
    }
    in_arrays.push_back(static_cast<InsightArray *>(inputs[2 + i]));
  }

  // Extract axis
  if (!inputs[2 + num_inputs]) {
    gpu_set_last_error("concat: axis is null");
    return C_FAILED;
  }
  int axis = *static_cast<int *>(inputs[2 + num_inputs]);

  int64_t ndim = out->ndim;
  int64_t element_size = 1;
  switch (out->dtype) {
  case INSIGHT_DTYPE_BOOL:
  case INSIGHT_DTYPE_U8:
  case INSIGHT_DTYPE_I8:
    element_size = 1;
    break;
  case INSIGHT_DTYPE_I16:
  case INSIGHT_DTYPE_U16:
    element_size = 2;
    break;
  case INSIGHT_DTYPE_I32:
  case INSIGHT_DTYPE_U32:
  case INSIGHT_DTYPE_F32:
    element_size = 4;
    break;
  case INSIGHT_DTYPE_I64:
  case INSIGHT_DTYPE_U64:
  case INSIGHT_DTYPE_F64:
  case INSIGHT_DTYPE_C32:
    element_size = 8;
    break;
  case INSIGHT_DTYPE_C64:
    element_size = 16;
    break;
  default:
    gpu_set_last_error("concat: unsupported dtype");
    return C_FAILED;
  }

  // Compute output strides
  int64_t out_strides[INSIGHT_MAX_NDIM];
  out_strides[ndim - 1] = 1;
  for (int d = ndim - 2; d >= 0; --d) {
    out_strides[d] = out_strides[d + 1] * out->dims[d + 1];
  }

  // Copy data from each input to the appropriate region of output
  int64_t axis_offset = 0;
  for (int i = 0; i < num_inputs; ++i) {
    InsightArray *src = in_arrays[i];
    int64_t src_size = src->numel;

    // Compute source strides
    int64_t src_strides[INSIGHT_MAX_NDIM];
    src_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
      src_strides[d] = src_strides[d + 1] * src->dims[d + 1];
    }

    // Copy element by element (simplified version)
    for (int64_t linear = 0; linear < src_size; ++linear) {
      // Convert linear index to multi-dimensional indices
      int64_t indices[INSIGHT_MAX_NDIM];
      int64_t remaining = linear;
      for (int d = ndim - 1; d >= 0; --d) {
        indices[d] = remaining % src->dims[d];
        remaining /= src->dims[d];
      }

      // Adjust axis index for output
      indices[axis] += axis_offset;

      // Compute output linear index
      int64_t out_linear = 0;
      for (int d = 0; d < ndim; ++d) {
        out_linear += indices[d] * out_strides[d];
      }

      // Copy element
      cudaMemcpy(static_cast<char *>(out->data) + out_linear * element_size,
                 static_cast<const char *>(src->data) + linear * element_size,
                 element_size, cudaMemcpyDeviceToDevice);
    }

    axis_offset += src->dims[axis];
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

// Register for all supported types
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_BOOL, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_U8, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_I8, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_I16, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_I32, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_I64, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_U16, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_U32, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_U64, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_F32, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_F64, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_C32, concat_kernel_gpu);
REGISTER_GPU_KERNEL(concat, INSIGHT_DTYPE_C64, concat_kernel_gpu);
