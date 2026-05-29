// backends/cuda/kernels/indexing/scatter_reduce.cu
/**
 * @file scatter_reduce.cu
 * @brief CUDA kernel for scatter_reduce operation.
 */

#include "../../registry/cuda_registry.h"
#include "../common/atomic_helpers.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void scatter_add_kernel(T *x, const int64_t *indices,
                                   const T *values, int64_t n,
                                   int64_t index_size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t batch = idx / index_size;
    int64_t i = idx % index_size;
    if constexpr (std::is_same_v<T, double>) {
      atomicAddDouble(&x[batch * index_size + indices[i]], values[idx]);
    } else {
      atomicAdd(&x[batch * index_size + indices[i]], values[idx]);
    }
  }
}

extern "C" {

C_Status scatter_reduce_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  InsightArray *indices = static_cast<InsightArray *>(inputs[1]);
  InsightArray *values = static_cast<InsightArray *>(inputs[2]);

  if (!x || !indices || !values) {
    gpu_set_last_error("scatter_reduce: null array pointer");
    return C_FAILED;
  }

  int axis = *static_cast<int *>(inputs[3]);
  int reduce_type = *static_cast<int *>(inputs[4]);

  // Only support add for now
  if (reduce_type != 0) {
    gpu_set_last_error("scatter_reduce: only add reduction supported");
    return C_FAILED;
  }

  int64_t n = values->numel;
  int64_t index_size = indices->numel / (n / values->dims[axis]);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32:
    scatter_add_kernel<float><<<blocks, threads>>>(
        static_cast<float *>(x->data),
        static_cast<const int64_t *>(indices->data),
        static_cast<const float *>(values->data), n, index_size);
    break;
  case INSIGHT_DTYPE_F64:
    scatter_add_kernel<double><<<blocks, threads>>>(
        static_cast<double *>(x->data),
        static_cast<const int64_t *>(indices->data),
        static_cast<const double *>(values->data), n, index_size);
    break;
  default:
    gpu_set_last_error("scatter_reduce: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_GPU_KERNEL(scatter_reduce, INSIGHT_DTYPE_F32,
                    scatter_reduce_kernel_gpu);
REGISTER_GPU_KERNEL(scatter_reduce, INSIGHT_DTYPE_F64,
                    scatter_reduce_kernel_gpu);
