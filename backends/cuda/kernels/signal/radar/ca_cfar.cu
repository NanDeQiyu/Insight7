// backends/cuda/kernels/signal/radar/ca_cfar.cu
// CUDA kernel for Cell-Averaging CFAR detection.
// 1D: each thread handles one cell. Prefix sum computed on device.
// 2D: each thread handles one cell using 2D prefix sum.
#include "../../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cuda_runtime.h>

// 1D CA-CFAR kernel
__global__ void ca_cfar_1d_f64(const double *src, double *th, bool *det,
                               const double *cumsum, int64_t n, double alpha,
                               int g, int r) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int64_t left_start = max((int64_t)0, i - g - r);
  int64_t left_end = max((int64_t)0, i - g);
  int64_t right_start = min(n, i + g + 1);
  int64_t right_end = min(n, i + g + r + 1);
  int64_t count = (left_end - left_start) + (right_end - right_start);
  if (count <= 0) {
    th[i] = 0;
    det[i] = false;
    return;
  }
  double sum_lr = (cumsum[left_end] - cumsum[left_start]) +
                  (cumsum[right_end] - cumsum[right_start]);
  double noise_level = sum_lr / count;
  th[i] = noise_level * alpha;
  det[i] = src[i] > th[i];
}

__global__ void ca_cfar_1d_f32(const float *src, float *th, bool *det,
                               const float *cumsum, int64_t n, float alpha,
                               int g, int r) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int64_t left_start = max((int64_t)0, i - g - r);
  int64_t left_end = max((int64_t)0, i - g);
  int64_t right_start = min(n, i + g + 1);
  int64_t right_end = min(n, i + g + r + 1);
  int64_t count = (left_end - left_start) + (right_end - right_start);
  if (count <= 0) {
    th[i] = 0;
    det[i] = false;
    return;
  }
  float sum_lr = (cumsum[left_end] - cumsum[left_start]) +
                 (cumsum[right_end] - cumsum[right_start]);
  float noise_level = sum_lr / count;
  th[i] = noise_level * alpha;
  det[i] = src[i] > th[i];
}

// 2D CA-CFAR kernel — each thread handles one cell
__global__ void ca_cfar_2d_f64(const double *src, double *th, bool *det,
                               const double *cs, int64_t rows, int64_t cols,
                               double alpha, int gr, int gc, int rr, int rc) {
  int64_t row = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= cols)
    return;

  int64_t cs_cols = cols + 1;
  int64_t r0 = max((int64_t)0, row - gr - rr);
  int64_t r1 = max((int64_t)0, row - gr);
  int64_t r2 = min(rows, row + gr + 1);
  int64_t r3 = min(rows, row + gr + rr + 1);
  int64_t c0 = max((int64_t)0, col - gc - rc);
  int64_t c1 = max((int64_t)0, col - gc);
  int64_t c2 = min(cols, col + gc + 1);
  int64_t c3 = min(cols, col + gc + rc + 1);

  double outer = cs[r3 * cs_cols + c3] - cs[r0 * cs_cols + c3] -
                 cs[r3 * cs_cols + c0] + cs[r0 * cs_cols + c0];
  double inner = cs[r2 * cs_cols + c2] - cs[r1 * cs_cols + c2] -
                 cs[r2 * cs_cols + c1] + cs[r1 * cs_cols + c1];
  double ref_sum = outer - inner;
  int64_t ref_count = ((r3 - r0) * (c3 - c0)) - ((r2 - r1) * (c2 - c1));

  int64_t idx = row * cols + col;
  if (ref_count <= 0) {
    th[idx] = 0;
    det[idx] = false;
    return;
  }
  double noise_level = ref_sum / ref_count;
  th[idx] = noise_level * alpha;
  det[idx] = src[idx] > th[idx];
}

__global__ void ca_cfar_2d_f32(const float *src, float *th, bool *det,
                               const float *cs, int64_t rows, int64_t cols,
                               float alpha, int gr, int gc, int rr, int rc) {
  int64_t row = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= cols)
    return;

  int64_t cs_cols = cols + 1;
  int64_t r0 = max((int64_t)0, row - gr - rr);
  int64_t r1 = max((int64_t)0, row - gr);
  int64_t r2 = min(rows, row + gr + 1);
  int64_t r3 = min(rows, row + gr + rr + 1);
  int64_t c0 = max((int64_t)0, col - gc - rc);
  int64_t c1 = max((int64_t)0, col - gc);
  int64_t c2 = min(cols, col + gc + 1);
  int64_t c3 = min(cols, col + gc + rc + 1);

  float outer = cs[r3 * cs_cols + c3] - cs[r0 * cs_cols + c3] -
                cs[r3 * cs_cols + c0] + cs[r0 * cs_cols + c0];
  float inner = cs[r2 * cs_cols + c2] - cs[r1 * cs_cols + c2] -
                cs[r2 * cs_cols + c1] + cs[r1 * cs_cols + c1];
  float ref_sum = outer - inner;
  int64_t ref_count = ((r3 - r0) * (c3 - c0)) - ((r2 - r1) * (c2 - c1));

  int64_t idx = row * cols + col;
  if (ref_count <= 0) {
    th[idx] = 0;
    det[idx] = false;
    return;
  }
  float noise_level = ref_sum / ref_count;
  th[idx] = noise_level * alpha;
  det[idx] = src[idx] > th[idx];
}

// Inclusive scan kernel for prefix sum (simple, not optimized)
__global__ void inclusive_scan_f64(const double *in, double *out, int64_t n) {
  // Use sequential scan per block + inter-block propagation
  // For simplicity, single-threaded scan (adequate for signal processing sizes)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = in[0];
    for (int64_t i = 1; i < n; ++i) {
      out[i] = out[i - 1] + in[i];
    }
  }
}

__global__ void inclusive_scan_f32(const float *in, float *out, int64_t n) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = in[0];
    for (int64_t i = 1; i < n; ++i) {
      out[i] = out[i - 1] + in[i];
    }
  }
}

extern "C" {

C_Status ca_cfar_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *data = (InsightArray *)inputs[0];
  InsightArray *alpha_arr = (InsightArray *)inputs[1];
  InsightArray *gc_arr = (InsightArray *)inputs[2];
  InsightArray *rc_arr = (InsightArray *)inputs[3];
  InsightArray *th = (InsightArray *)outputs[0];
  InsightArray *det = (InsightArray *)outputs[1];

  if (!data || !alpha_arr || !gc_arr || !rc_arr || !th || !det) {
    gpu_set_last_error("ca_cfar: null array pointer");
    return C_FAILED;
  }

  double alpha = *(double *)alpha_arr->data;
  int g = *(int *)gc_arr->data;
  int r = *(int *)rc_arr->data;

  if (data->ndim == 1) {
    int64_t n = data->dims[0];
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    // Compute prefix sum on device
    void *cumsum_ptr = nullptr;
    cudaMalloc(&cumsum_ptr, (n + 1) * (data->dtype == INSIGHT_DTYPE_F64
                                            ? sizeof(double)
                                            : sizeof(float)));

    if (data->dtype == INSIGHT_DTYPE_F64) {
      // Build cumsum array (shifted by 1, cumsum[0]=0)
      // For simplicity, use a host copy then device scan
      inclusive_scan_f64<<<1, 1>>>((const double *)data->data,
                                   (double *)cumsum_ptr + 1, n);
      cudaMemset(cumsum_ptr, 0, sizeof(double));
      ca_cfar_1d_f64<<<blocks, threads>>>(
          (const double *)data->data, (double *)th->data, (bool *)det->data,
          (const double *)cumsum_ptr, n, alpha, g, r);
    } else {
      inclusive_scan_f32<<<1, 1>>>((const float *)data->data,
                                   (float *)cumsum_ptr + 1, n);
      cudaMemset(cumsum_ptr, 0, sizeof(float));
      ca_cfar_1d_f32<<<blocks, threads>>>(
          (const float *)data->data, (float *)th->data, (bool *)det->data,
          (const float *)cumsum_ptr, n, alpha, g, r);
    }
    cudaFree(cumsum_ptr);
  } else if (data->ndim == 2) {
    int64_t rows = data->dims[0];
    int64_t cols = data->dims[1];

    int gc_col = gc_arr->numel > 1 ? ((int *)gc_arr->data)[1] : g;
    int rc_col = rc_arr->numel > 1 ? ((int *)rc_arr->data)[1] : r;

    dim3 threads(16, 16);
    dim3 blocks_2d((int)((cols + 15) / 16), (int)((rows + 15) / 16));

    // 2D prefix sum computed on host then copied (simple approach)
    // For production, use thrust::inclusive_scan_by_key or custom kernel
    int64_t cs_size = (rows + 1) * (cols + 1);

    if (data->dtype == INSIGHT_DTYPE_F64) {
      // Allocate device memory for 2D cumsum
      double *cs_dev = nullptr;
      cudaMalloc(&cs_dev, cs_size * sizeof(double));
      cudaMemset(cs_dev, 0, cs_size * sizeof(double));

      // Compute 2D prefix sum on host (simple, correct)
      std::vector<double> src_h(rows * cols), cs_h(cs_size, 0.0);
      cudaMemcpy(src_h.data(), data->data, rows * cols * sizeof(double),
                 cudaMemcpyDeviceToHost);
      for (int64_t rr = 0; rr < rows; ++rr) {
        for (int64_t cc = 0; cc < cols; ++cc) {
          cs_h[(rr + 1) * (cols + 1) + (cc + 1)] =
              src_h[rr * cols + cc] + cs_h[rr * (cols + 1) + (cc + 1)] +
              cs_h[(rr + 1) * (cols + 1) + cc] - cs_h[rr * (cols + 1) + cc];
        }
      }
      cudaMemcpy(cs_dev, cs_h.data(), cs_size * sizeof(double),
                 cudaMemcpyHostToDevice);

      ca_cfar_2d_f64<<<blocks_2d, threads>>>(
          (const double *)data->data, (double *)th->data, (bool *)det->data,
          cs_dev, rows, cols, alpha, g, gc_col, r, rc_col);
      cudaFree(cs_dev);
    } else {
      float *cs_dev = nullptr;
      cudaMalloc(&cs_dev, cs_size * sizeof(float));
      cudaMemset(cs_dev, 0, cs_size * sizeof(float));

      std::vector<float> src_h(rows * cols), cs_h(cs_size, 0.0f);
      cudaMemcpy(src_h.data(), data->data, rows * cols * sizeof(float),
                 cudaMemcpyDeviceToHost);
      for (int64_t rr = 0; rr < rows; ++rr) {
        for (int64_t cc = 0; cc < cols; ++cc) {
          cs_h[(rr + 1) * (cols + 1) + (cc + 1)] =
              src_h[rr * cols + cc] + cs_h[rr * (cols + 1) + (cc + 1)] +
              cs_h[(rr + 1) * (cols + 1) + cc] - cs_h[rr * (cols + 1) + cc];
        }
      }
      cudaMemcpy(cs_dev, cs_h.data(), cs_size * sizeof(float),
                 cudaMemcpyHostToDevice);

      ca_cfar_2d_f32<<<blocks_2d, threads>>>(
          (const float *)data->data, (float *)th->data, (bool *)det->data,
          cs_dev, rows, cols, alpha, g, gc_col, r, rc_col);
      cudaFree(cs_dev);
    }
  } else {
    gpu_set_last_error("ca_cfar: supports 1D and 2D arrays only");
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

REGISTER_GPU_KERNEL(ca_cfar, INSIGHT_DTYPE_F32, ca_cfar_kernel_gpu);
REGISTER_GPU_KERNEL(ca_cfar, INSIGHT_DTYPE_F64, ca_cfar_kernel_gpu);
