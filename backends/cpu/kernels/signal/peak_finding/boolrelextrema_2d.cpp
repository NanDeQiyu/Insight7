// backends/cpu/kernels/signal/peak_finding/boolrelextrema_2d.cpp
// CPU kernel for 2D relative extrema detection
// Operates along the last axis (columns) for each row.
// For each element, checks if it is a local max/min by comparing with
// 'order' neighbors on each side along the specified axis.
// inputs: [0]=data (2D, rows x cols), [1]=order (I32 scalar),
//         [2]=comparator (I32: 0=greater, 1=less),
//         [3]=axis (I32: 0=along rows, 1=along cols)
// outputs: [0]=mask (bool, same shape as data)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status boolrelextrema_2d_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *data = (InsightArray *)inputs[0];
  int32_t order = *(int32_t *)((InsightArray *)inputs[1])->data;
  int32_t cmp = *(int32_t *)((InsightArray *)inputs[2])->data;
  int32_t axis = *(int32_t *)((InsightArray *)inputs[3])->data;
  InsightArray *out = (InsightArray *)outputs[0];

  if (!data || !out) {
    cpu_set_last_error("boolrelextrema_2d: null array pointer");
    return C_FAILED;
  }

  int64_t rows = data->dims[0];
  int64_t cols = data->dims[1];

  auto check_extremum = [&](double val, double neighbor) -> bool {
    if (cmp == 0)
      return val < neighbor; // greater: fail if neighbor is larger
    return val > neighbor;   // less: fail if neighbor is smaller
  };

  if (data->dtype == INSIGHT_DTYPE_F64) {
    const double *dd = (const double *)data->data;
    bool *od = (bool *)out->data;

    if (axis == 1) {
      // Along columns (default): for each row, scan columns
#pragma omp parallel for if (rows > 4)
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          bool is_extremum = true;
          double val = dd[r * cols + c];
          for (int64_t j = 1; j <= order; ++j) {
            if (c - j >= 0 && check_extremum(val, dd[r * cols + c - j])) {
              is_extremum = false;
              break;
            }
            if (c + j < cols && check_extremum(val, dd[r * cols + c + j])) {
              is_extremum = false;
              break;
            }
          }
          od[r * cols + c] = is_extremum;
        }
      }
    } else {
      // Along rows: for each column, scan rows
#pragma omp parallel for if (cols > 4)
      for (int64_t c = 0; c < cols; ++c) {
        for (int64_t r = 0; r < rows; ++r) {
          bool is_extremum = true;
          double val = dd[r * cols + c];
          for (int64_t j = 1; j <= order; ++j) {
            if (r - j >= 0 && check_extremum(val, dd[(r - j) * cols + c])) {
              is_extremum = false;
              break;
            }
            if (r + j < rows && check_extremum(val, dd[(r + j) * cols + c])) {
              is_extremum = false;
              break;
            }
          }
          od[r * cols + c] = is_extremum;
        }
      }
    }
  } else if (data->dtype == INSIGHT_DTYPE_F32) {
    const float *dd = (const float *)data->data;
    bool *od = (bool *)out->data;

    auto check_extremum_f32 = [&](float val, float neighbor) -> bool {
      if (cmp == 0)
        return val < neighbor;
      return val > neighbor;
    };

    if (axis == 1) {
#pragma omp parallel for if (rows > 4)
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          bool is_extremum = true;
          float val = dd[r * cols + c];
          for (int64_t j = 1; j <= order; ++j) {
            if (c - j >= 0 && check_extremum_f32(val, dd[r * cols + c - j])) {
              is_extremum = false;
              break;
            }
            if (c + j < cols && check_extremum_f32(val, dd[r * cols + c + j])) {
              is_extremum = false;
              break;
            }
          }
          od[r * cols + c] = is_extremum;
        }
      }
    } else {
#pragma omp parallel for if (cols > 4)
      for (int64_t c = 0; c < cols; ++c) {
        for (int64_t r = 0; r < rows; ++r) {
          bool is_extremum = true;
          float val = dd[r * cols + c];
          for (int64_t j = 1; j <= order; ++j) {
            if (r - j >= 0 && check_extremum_f32(val, dd[(r - j) * cols + c])) {
              is_extremum = false;
              break;
            }
            if (r + j < rows &&
                check_extremum_f32(val, dd[(r + j) * cols + c])) {
              is_extremum = false;
              break;
            }
          }
          od[r * cols + c] = is_extremum;
        }
      }
    }
  } else {
    cpu_set_last_error("boolrelextrema_2d: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(boolrelextrema_2d, INSIGHT_DTYPE_F64,
                    boolrelextrema_2d_kernel_cpu);
REGISTER_CPU_KERNEL(boolrelextrema_2d, INSIGHT_DTYPE_F32,
                    boolrelextrema_2d_kernel_cpu);
