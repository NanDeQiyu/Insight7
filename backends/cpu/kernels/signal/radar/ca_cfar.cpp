// backends/cpu/kernels/signal/radar/ca_cfar.cpp
// CPU kernel for Cell-Averaging CFAR detection
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" {

// ca_cfar kernel — 1D and 2D
// inputs: [0]=data, [1]=alpha (scalar f64), [2]=guard_cells (int32),
// [3]=reference_cells (int32) outputs: [0]=threshold, [1]=detections (bool)
C_Status ca_cfar_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *data = (InsightArray *)inputs[0];
  InsightArray *alpha_arr = (InsightArray *)inputs[1];
  InsightArray *gc_arr = (InsightArray *)inputs[2];
  InsightArray *rc_arr = (InsightArray *)inputs[3];
  InsightArray *th = (InsightArray *)outputs[0];
  InsightArray *det = (InsightArray *)outputs[1];

  if (!data || !alpha_arr || !gc_arr || !rc_arr || !th || !det) {
    cpu_set_last_error("ca_cfar: null array pointer");
    return C_FAILED;
  }

  double alpha = *(double *)alpha_arr->data;
  int g = *(int *)gc_arr->data;
  int r = *(int *)rc_arr->data;

  if (data->ndim == 1) {
    int64_t n = data->dims[0];

    if (data->dtype == INSIGHT_DTYPE_F64) {
      const double *src = (const double *)data->data;
      double *th_data = (double *)th->data;
      bool *det_data = (bool *)det->data;

      std::vector<double> cumsum(n + 1, 0.0);
      for (int64_t i = 0; i < n; ++i)
        cumsum[i + 1] = cumsum[i] + src[i];

      for (int64_t i = 0; i < n; ++i) {
        int64_t left_start = std::max((int64_t)0, i - g - r);
        int64_t left_end = std::max((int64_t)0, i - g);
        int64_t right_start = std::min(n, i + g + 1);
        int64_t right_end = std::min(n, i + g + r + 1);
        int64_t count = (left_end - left_start) + (right_end - right_start);
        if (count == 0) {
          th_data[i] = 0;
          det_data[i] = false;
          continue;
        }
        double sum_lr = (cumsum[left_end] - cumsum[left_start]) +
                        (cumsum[right_end] - cumsum[right_start]);
        double noise_level = sum_lr / count;
        th_data[i] = noise_level * alpha;
        det_data[i] = src[i] > th_data[i];
      }
    } else {
      const float *src = (const float *)data->data;
      float *th_data = (float *)th->data;
      bool *det_data = (bool *)det->data;

      std::vector<float> cumsum(n + 1, 0.0f);
      for (int64_t i = 0; i < n; ++i)
        cumsum[i + 1] = cumsum[i] + src[i];

      for (int64_t i = 0; i < n; ++i) {
        int64_t left_start = std::max((int64_t)0, i - g - r);
        int64_t left_end = std::max((int64_t)0, i - g);
        int64_t right_start = std::min(n, i + g + 1);
        int64_t right_end = std::min(n, i + g + r + 1);
        int64_t count = (left_end - left_start) + (right_end - right_start);
        if (count == 0) {
          th_data[i] = 0;
          det_data[i] = false;
          continue;
        }
        float sum_lr = (cumsum[left_end] - cumsum[left_start]) +
                       (cumsum[right_end] - cumsum[right_start]);
        float noise_level = sum_lr / count;
        th_data[i] = noise_level * static_cast<float>(alpha);
        det_data[i] = src[i] > th_data[i];
      }
    }
  } else if (data->ndim == 2) {
    int64_t rows = data->dims[0];
    int64_t cols = data->dims[1];

    // For 2D, guard_cells and reference_cells may have 2 elements
    int gc_col = gc_arr->numel > 1 ? ((int *)gc_arr->data)[1] : g;
    int rc_col = rc_arr->numel > 1 ? ((int *)rc_arr->data)[1] : r;
    int gr = g, gc = gc_col, rr = r, rc = rc_col;

    if (data->dtype == INSIGHT_DTYPE_F64) {
      const double *src = (const double *)data->data;
      double *th_data = (double *)th->data;
      bool *det_data = (bool *)det->data;

      std::vector<double> cs((rows + 1) * (cols + 1), 0.0);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          cs[(r + 1) * (cols + 1) + (c + 1)] =
              src[r * cols + c] + cs[r * (cols + 1) + (c + 1)] +
              cs[(r + 1) * (cols + 1) + c] - cs[r * (cols + 1) + c];
        }
      }

      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
          int64_t r0 = std::max((int64_t)0, row - gr - rr);
          int64_t r1 = std::max((int64_t)0, row - gr);
          int64_t r2 = std::min(rows, row + gr + 1);
          int64_t r3 = std::min(rows, row + gr + rr + 1);
          int64_t c0 = std::max((int64_t)0, col - gc - rc);
          int64_t c1 = std::max((int64_t)0, col - gc);
          int64_t c2 = std::min(cols, col + gc + 1);
          int64_t c3 = std::min(cols, col + gc + rc + 1);

          double outer = cs[r3 * (cols + 1) + c3] - cs[r0 * (cols + 1) + c3] -
                         cs[r3 * (cols + 1) + c0] + cs[r0 * (cols + 1) + c0];
          double inner = cs[r2 * (cols + 1) + c2] - cs[r1 * (cols + 1) + c2] -
                         cs[r2 * (cols + 1) + c1] + cs[r1 * (cols + 1) + c1];
          double ref_sum = outer - inner;
          int64_t ref_count = ((r3 - r0) * (c3 - c0)) - ((r2 - r1) * (c2 - c1));
          if (ref_count <= 0) {
            th_data[row * cols + col] = 0;
            det_data[row * cols + col] = false;
            continue;
          }
          double noise_level = ref_sum / ref_count;
          th_data[row * cols + col] = noise_level * alpha;
          det_data[row * cols + col] =
              src[row * cols + col] > th_data[row * cols + col];
        }
      }
    } else {
      const float *src = (const float *)data->data;
      float *th_data = (float *)th->data;
      bool *det_data = (bool *)det->data;

      std::vector<float> cs((rows + 1) * (cols + 1), 0.0f);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          cs[(r + 1) * (cols + 1) + (c + 1)] =
              src[r * cols + c] + cs[r * (cols + 1) + (c + 1)] +
              cs[(r + 1) * (cols + 1) + c] - cs[r * (cols + 1) + c];
        }
      }

      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
          int64_t r0 = std::max((int64_t)0, row - gr - rr);
          int64_t r1 = std::max((int64_t)0, row - gr);
          int64_t r2 = std::min(rows, row + gr + 1);
          int64_t r3 = std::min(rows, row + gr + rr + 1);
          int64_t c0 = std::max((int64_t)0, col - gc - rc);
          int64_t c1 = std::max((int64_t)0, col - gc);
          int64_t c2 = std::min(cols, col + gc + 1);
          int64_t c3 = std::min(cols, col + gc + rc + 1);

          float outer = cs[r3 * (cols + 1) + c3] - cs[r0 * (cols + 1) + c3] -
                        cs[r3 * (cols + 1) + c0] + cs[r0 * (cols + 1) + c0];
          float inner = cs[r2 * (cols + 1) + c2] - cs[r1 * (cols + 1) + c2] -
                        cs[r2 * (cols + 1) + c1] + cs[r1 * (cols + 1) + c1];
          float ref_sum = outer - inner;
          int64_t ref_count = ((r3 - r0) * (c3 - c0)) - ((r2 - r1) * (c2 - c1));
          if (ref_count <= 0) {
            th_data[row * cols + col] = 0;
            det_data[row * cols + col] = false;
            continue;
          }
          float noise_level = ref_sum / ref_count;
          th_data[row * cols + col] = noise_level * static_cast<float>(alpha);
          det_data[row * cols + col] =
              src[row * cols + col] > th_data[row * cols + col];
        }
      }
    }
  } else {
    cpu_set_last_error("ca_cfar: supports 1D and 2D arrays only");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(ca_cfar, INSIGHT_DTYPE_F64, ca_cfar_kernel_cpu);
REGISTER_CPU_KERNEL(ca_cfar, INSIGHT_DTYPE_F32, ca_cfar_kernel_cpu);
