// backends/cuda/kernels/signal/windows/kaiser.cu
// CUDA kernel for Kaiser window generation.
// w[n] = I0(beta * sqrt(1 - ((n - (M-1)/2) / ((M-1)/2))^2)) / I0(beta)
// where I0 is the modified Bessel function of the first kind, order 0.
// inputs[0]: beta (1-element F64 array on device)
// outputs[0]: window output
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_runtime.h>

// Modified Bessel function I0(x) using Abramowitz & Stegun approximation.
// Accurate to ~1.6e-7 for all x >= 0.
__device__ double bessel_i0_dev(double x) {
  double ax = fabs(x);
  if (ax <= 3.75) {
    double t = x / 3.75;
    double t2 = t * t;
    return 1.0 + 3.5156229 * t2 + 3.0899424 * t2 * t2 +
           1.2067492 * t2 * t2 * t2 + 0.2659732 * t2 * t2 * t2 * t2 +
           0.0360768 * t2 * t2 * t2 * t2 * t2 +
           0.0045813 * t2 * t2 * t2 * t2 * t2 * t2;
  } else {
    double t = 3.75 / ax;
    return (exp(ax) / sqrt(ax)) *
           (0.39894228 + 0.01328592 * t + 0.000225319 * t * t -
            0.00157565 * t * t * t + 0.00916281 * t * t * t * t -
            0.02057706 * t * t * t * t * t +
            0.02635537 * t * t * t * t * t * t -
            0.01647633 * t * t * t * t * t * t * t +
            0.00392377 * t * t * t * t * t * t * t * t);
  }
}

template <typename T>
__global__ void kaiser_kernel(T *out, int64_t M, T beta, T i0_beta) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  T half = T(M - 1) / T(2.0);
  T inv_half = (half > T(0.0)) ? T(1.0) / half : T(0.0);
  T ratio = (T(i) - half) * inv_half;
  T inner = T(1.0) - ratio * ratio;
  T arg = beta * sqrt(fmax(T(0.0), inner));
  out[i] = bessel_i0_dev(arg) / i0_beta;
}

extern "C" {

C_Status kaiser_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *beta_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!beta_arr || !out) {
    gpu_set_last_error("kaiser: null array pointer");
    return C_FAILED;
  }

  // Read beta from device memory
  double beta_host;
  cudaMemcpy(&beta_host, beta_arr->data, sizeof(double),
             cudaMemcpyDeviceToHost);

  // Compute I0(beta) on host
  // Host-side bessel_i0 (same algorithm as device version)
  double ax = fabs(beta_host);
  double i0_beta;
  if (ax <= 3.75) {
    double t = beta_host / 3.75;
    double t2 = t * t;
    i0_beta = 1.0 + 3.5156229 * t2 + 3.0899424 * t2 * t2 +
              1.2067492 * t2 * t2 * t2 + 0.2659732 * t2 * t2 * t2 * t2 +
              0.0360768 * t2 * t2 * t2 * t2 * t2 +
              0.0045813 * t2 * t2 * t2 * t2 * t2 * t2;
  } else {
    double t = 3.75 / ax;
    i0_beta =
        (exp(ax) / sqrt(ax)) *
        (0.39894228 + 0.01328592 * t + 0.000225319 * t * t -
         0.00157565 * t * t * t + 0.00916281 * t * t * t * t -
         0.02057706 * t * t * t * t * t + 0.02635537 * t * t * t * t * t * t -
         0.01647633 * t * t * t * t * t * t * t +
         0.00392377 * t * t * t * t * t * t * t * t);
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  dim3 blocks((int)((M + 255) / 256));
  dim3 threads(256);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F64:
    kaiser_kernel<double>
        <<<blocks, threads>>>((double *)out->data, M, beta_host, i0_beta);
    break;
  default:
    gpu_set_last_error("kaiser: only F64 dtype supported");
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

REGISTER_GPU_KERNEL(kaiser, INSIGHT_DTYPE_F64, kaiser_kernel_gpu);
