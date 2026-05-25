// backends/cuda/kernels/creation/full.cu
#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <complex>
#include <cuda_runtime.h>
#include <string>

template <typename T> __global__ void full_kernel(T *dst, T val, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = val;
  }
}

extern "C" {

C_Status full_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("full_kernel_gpu: output array is null");
    return C_FAILED;
  }
  if (!inputs[1]) {
    gpu_set_last_error("full_kernel_gpu: fill_value is null");
    return C_FAILED;
  }

  double fill_val = *static_cast<double *>(inputs[1]);
  int64_t n = out->numel;
  int32_t dtype = out->dtype;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  switch (dtype) {
  case INSIGHT_DTYPE_BOOL: {
    bool val = (fill_val != 0.0);
    full_kernel<<<blocks, threads>>>(static_cast<bool *>(out->data), val, n);
    break;
  }
  case INSIGHT_DTYPE_U8: {
    uint8_t val = static_cast<uint8_t>(fill_val);
    full_kernel<<<blocks, threads>>>(static_cast<uint8_t *>(out->data), val, n);
    break;
  }
  case INSIGHT_DTYPE_I8: {
    int8_t val = static_cast<int8_t>(fill_val);
    full_kernel<<<blocks, threads>>>(static_cast<int8_t *>(out->data), val, n);
    break;
  }
  case INSIGHT_DTYPE_I16: {
    int16_t val = static_cast<int16_t>(fill_val);
    full_kernel<<<blocks, threads>>>(static_cast<int16_t *>(out->data), val, n);
    break;
  }
  case INSIGHT_DTYPE_I32: {
    int32_t val = static_cast<int32_t>(fill_val);
    full_kernel<<<blocks, threads>>>(static_cast<int32_t *>(out->data), val, n);
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t val = static_cast<int64_t>(fill_val);
    full_kernel<<<blocks, threads>>>(static_cast<int64_t *>(out->data), val, n);
    break;
  }
  case INSIGHT_DTYPE_U16: {
    uint16_t val = static_cast<uint16_t>(fill_val);
    full_kernel<<<blocks, threads>>>(static_cast<uint16_t *>(out->data), val,
                                     n);
    break;
  }
  case INSIGHT_DTYPE_U32: {
    uint32_t val = static_cast<uint32_t>(fill_val);
    full_kernel<<<blocks, threads>>>(static_cast<uint32_t *>(out->data), val,
                                     n);
    break;
  }
  case INSIGHT_DTYPE_U64: {
    uint64_t val = static_cast<uint64_t>(fill_val);
    full_kernel<<<blocks, threads>>>(static_cast<uint64_t *>(out->data), val,
                                     n);
    break;
  }
  case INSIGHT_DTYPE_F32: {
    float val = static_cast<float>(fill_val);
    full_kernel<<<blocks, threads>>>(static_cast<float *>(out->data), val, n);
    break;
  }
  case INSIGHT_DTYPE_F64: {
    full_kernel<<<blocks, threads>>>(static_cast<double *>(out->data), fill_val,
                                     n);
    break;
  }
  case INSIGHT_DTYPE_C32: {
    std::complex<float> val(static_cast<float>(fill_val), 0.0f);
    full_kernel<<<blocks, threads>>>(
        static_cast<std::complex<float> *>(out->data), val, n);
    break;
  }
  case INSIGHT_DTYPE_C64: {
    std::complex<double> val(fill_val, 0.0);
    full_kernel<<<blocks, threads>>>(
        static_cast<std::complex<double> *>(out->data), val, n);
    break;
  }
  default:
    gpu_set_last_error(
        ("full_kernel_gpu: unsupported dtype " + std::to_string(dtype))
            .c_str());
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

REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_BOOL, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_U8, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_I8, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_I16, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_I32, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_I64, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_U16, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_U32, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_U64, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_F32, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_F64, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_C32, full_kernel_gpu);
REGISTER_GPU_KERNEL(full, INSIGHT_DTYPE_C64, full_kernel_gpu);