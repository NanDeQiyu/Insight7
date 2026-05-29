// backends/cuda/kernels/manipulation/flip.cu
/**
 * @file flip.cu
 * @brief CUDA kernel for flip operation.
 */

#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cstring>
#include <cuda_runtime.h>

template <typename T>
__global__ void flip_kernel(const T *src, T *dst, int64_t n, int64_t ndim,
                            const int64_t *dims, const int64_t *src_strides,
                            const int64_t *dst_strides, int64_t axis,
                            int64_t axis_size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Convert linear index to multi-dimensional indices
    int64_t indices[10];
    int64_t remaining = idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
      indices[d] = remaining % dims[d];
      remaining /= dims[d];
    }

    // Flip the target axis
    indices[axis] = axis_size - 1 - indices[axis];

    // Compute source offset
    int64_t src_offset = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      src_offset += indices[d] * src_strides[d];
    }

    dst[idx] = src[src_offset];
  }
}

extern "C" {

C_Status flip_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);

  if (!out || !x) {
    gpu_set_last_error("flip: null array pointer");
    return C_FAILED;
  }

  int axis = *static_cast<int *>(inputs[2]);
  int64_t ndim = x->ndim;
  if (axis < 0)
    axis += ndim;

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  int64_t *d_dims, *d_src_strides, *d_dst_strides;
  cudaMalloc(&d_dims, ndim * sizeof(int64_t));
  cudaMalloc(&d_src_strides, ndim * sizeof(int64_t));
  cudaMalloc(&d_dst_strides, ndim * sizeof(int64_t));
  cudaMemcpy(d_dims, x->dims, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src_strides, x->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  // Compute contiguous strides for output
  int64_t dst_strides[10];
  dst_strides[ndim - 1] = 1;
  for (int d = ndim - 2; d >= 0; --d) {
    dst_strides[d] = dst_strides[d + 1] * out->dims[d + 1];
  }
  cudaMemcpy(d_dst_strides, dst_strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  int64_t axis_size = x->dims[axis];

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    flip_kernel<float><<<blocks, threads>>>(
        static_cast<const float *>(x->data), static_cast<float *>(out->data), n,
        ndim, d_dims, d_src_strides, d_dst_strides, axis, axis_size);
    break;
  case INSIGHT_DTYPE_F64:
    flip_kernel<double><<<blocks, threads>>>(
        static_cast<const double *>(x->data), static_cast<double *>(out->data),
        n, ndim, d_dims, d_src_strides, d_dst_strides, axis, axis_size);
    break;
  case INSIGHT_DTYPE_I32:
    flip_kernel<int32_t><<<blocks, threads>>>(
        static_cast<const int32_t *>(x->data),
        static_cast<int32_t *>(out->data), n, ndim, d_dims, d_src_strides,
        d_dst_strides, axis, axis_size);
    break;
  case INSIGHT_DTYPE_I64:
    flip_kernel<int64_t><<<blocks, threads>>>(
        static_cast<const int64_t *>(x->data),
        static_cast<int64_t *>(out->data), n, ndim, d_dims, d_src_strides,
        d_dst_strides, axis, axis_size);
    break;
  default:
    cudaFree(d_dims);
    cudaFree(d_src_strides);
    cudaFree(d_dst_strides);
    gpu_set_last_error("flip: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  cudaFree(d_dims);
  cudaFree(d_src_strides);
  cudaFree(d_dst_strides);

  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_GPU_KERNEL(flip, INSIGHT_DTYPE_F32, flip_kernel_gpu);
REGISTER_GPU_KERNEL(flip, INSIGHT_DTYPE_F64, flip_kernel_gpu);
REGISTER_GPU_KERNEL(flip, INSIGHT_DTYPE_I32, flip_kernel_gpu);
REGISTER_GPU_KERNEL(flip, INSIGHT_DTYPE_I64, flip_kernel_gpu);
