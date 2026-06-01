// backends/cpu/kernels/signal/convolution/convolve2d.cpp
// CPU kernel for 2D direct convolution
// y[i,j] = sum_{k,l} x[i+k-offset_r, j+l-offset_c] * h[k,l]
// mode: 0=valid, 1=same, 2=full
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status convolve2d_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *h = (InsightArray *)inputs[1];
  int32_t mode = *(int32_t *)((InsightArray *)inputs[2])->data;
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !h || !out) {
    cpu_set_last_error("convolve2d: null array pointer");
    return C_FAILED;
  }

  int64_t rows = x->dims[0];
  int64_t cols = x->dims[1];
  int64_t hrows = h->dims[0];
  int64_t hcols = h->dims[1];

  int64_t out_rows, out_cols, offset_r, offset_c;

  if (mode == 0) { // valid
    out_rows = rows - hrows + 1;
    out_cols = cols - hcols + 1;
    offset_r = 0;
    offset_c = 0;
  } else if (mode == 1) { // same
    out_rows = rows;
    out_cols = cols;
    offset_r = hrows / 2;
    offset_c = hcols / 2;
  } else { // full
    out_rows = rows + hrows - 1;
    out_cols = cols + hcols - 1;
    offset_r = 0;
    offset_c = 0;
  }

  if (x->dtype == INSIGHT_DTYPE_F64) {
    const double *xd = (const double *)x->data;
    const double *hd = (const double *)h->data;
    double *od = (double *)out->data;
#pragma omp parallel for if (out_rows > 4)
    for (int64_t i = 0; i < out_rows; ++i) {
      for (int64_t j = 0; j < out_cols; ++j) {
        double sum = 0.0;
        for (int64_t k = 0; k < hrows; ++k) {
          int64_t ri = i + k - offset_r;
          if (ri < 0 || ri >= rows)
            continue;
          for (int64_t l = 0; l < hcols; ++l) {
            int64_t ci = j + l - offset_c;
            if (ci >= 0 && ci < cols) {
              sum += xd[ri * cols + ci] * hd[k * hcols + l];
            }
          }
        }
        od[i * out_cols + j] = sum;
      }
    }
  } else if (x->dtype == INSIGHT_DTYPE_F32) {
    const float *xd = (const float *)x->data;
    const float *hd = (const float *)h->data;
    float *od = (float *)out->data;
#pragma omp parallel for if (out_rows > 4)
    for (int64_t i = 0; i < out_rows; ++i) {
      for (int64_t j = 0; j < out_cols; ++j) {
        float sum = 0.0f;
        for (int64_t k = 0; k < hrows; ++k) {
          int64_t ri = i + k - offset_r;
          if (ri < 0 || ri >= rows)
            continue;
          for (int64_t l = 0; l < hcols; ++l) {
            int64_t ci = j + l - offset_c;
            if (ci >= 0 && ci < cols) {
              sum += xd[ri * cols + ci] * hd[k * hcols + l];
            }
          }
        }
        od[i * out_cols + j] = sum;
      }
    }
  } else {
    cpu_set_last_error("convolve2d: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(convolve2d, INSIGHT_DTYPE_F64, convolve2d_kernel_cpu);
REGISTER_CPU_KERNEL(convolve2d, INSIGHT_DTYPE_F32, convolve2d_kernel_cpu);
