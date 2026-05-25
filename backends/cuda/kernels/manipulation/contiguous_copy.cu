// backends/cuda/kernels/manipulation/contiguous_copy.cu
#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

/**
 * @brief GPU kernel for copying a non-contiguous array to a contiguous one.
 *
 * Each thread copies one element, computing the source offset from
 * the input array's strides.
 */
__global__ void contiguous_copy_kernel(char *__restrict__ dst,
                                       const char *__restrict__ src,
                                       const int64_t *__restrict__ dims,
                                       const int64_t *__restrict__ strides,
                                       int64_t total, int ndim,
                                       size_t elem_size) {
  int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total)
    return;

  int64_t src_offset = 0;
  int64_t tmp = linear;
  for (int d = ndim - 1; d >= 0; --d) {
    int64_t idx = tmp % dims[d];
    src_offset += idx * strides[d];
    tmp /= dims[d];
  }

  const char *src_ptr = src + src_offset * elem_size;
  char *dst_ptr = dst + linear * elem_size;

  // Element-wise copy
  for (size_t b = 0; b < elem_size; ++b) {
    dst_ptr[b] = src_ptr[b];
  }
}

extern "C" {

C_Status contiguous_copy_gpu(void **inputs, void **outputs) {
  InsightArray *in = static_cast<InsightArray *>(inputs[0]);
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!in) {
    gpu_set_last_error("contiguous_copy: input array is null");
    return C_FAILED;
  }
  if (!out) {
    gpu_set_last_error("contiguous_copy: output array is null");
    return C_FAILED;
  }

  size_t elem_size = insight_dtype_size(in->dtype);
  if (elem_size == 0) {
    gpu_set_last_error("contiguous_copy: unknown data type");
    return C_FAILED;
  }

  int64_t total = in->numel;
  if (total == 0)
    return C_SUCCESS;

  // Fast path: if input is already contiguous, do direct device copy
  if (insight_array_is_contiguous(in)) {
    size_t bytes = total * elem_size;
    cudaError_t err =
        cudaMemcpy(out->data, in->data, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      gpu_set_last_error(cudaGetErrorString(err));
      return C_FAILED;
    }
    return C_SUCCESS;
  }

  // Prepare dims and strides on device
  int ndim = in->ndim;
  int64_t *d_dims;
  int64_t *d_strides;
  cudaMalloc(&d_dims, ndim * sizeof(int64_t));
  cudaMalloc(&d_strides, ndim * sizeof(int64_t));
  cudaMemcpy(d_dims, in->dims, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_strides, in->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  contiguous_copy_kernel<<<blocks, threads>>>(
      static_cast<char *>(out->data),
      static_cast<const char *>(in->data) + in->offset * elem_size, d_dims,
      d_strides, total, ndim, elem_size);

  cudaFree(d_dims);
  cudaFree(d_strides);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_BOOL, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_U8, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_I8, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_I16, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_I32, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_I64, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_U16, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_U32, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_U64, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_F16, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_BF16, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_F32, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_F64, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_C32, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_C64, contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_F8_E4M3,
                    contiguous_copy_gpu);
REGISTER_GPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_F8_E5M2,
                    contiguous_copy_gpu);