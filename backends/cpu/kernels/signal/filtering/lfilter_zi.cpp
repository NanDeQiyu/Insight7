// backends/cpu/kernels/signal/filtering/lfilter_zi.cpp
// CPU kernel for computing lfilter initial conditions (zi)
// Solves (I - A) * zi = b[1:] - a[1:] * b[0] via Gauss-Jordan elimination
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" {

// lfilter_zi kernel
// inputs: [0]=b_norm (normalized numerator), [1]=a_norm (normalized denominator)
// outputs: [0]=zi (initial conditions, length max(nb,na)-1)
//
// Both b_norm and a_norm are pre-normalized by a[0] and passed as double arrays.
C_Status lfilter_zi_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *b_arr = (InsightArray *)inputs[0];
  InsightArray *a_arr = (InsightArray *)inputs[1];
  InsightArray *zi_arr = (InsightArray *)outputs[0];

  if (!b_arr || !a_arr || !zi_arr) {
    cpu_set_last_error("lfilter_zi: null array pointer");
    return C_FAILED;
  }

  int64_t nb = b_arr->numel;
  int64_t na = a_arr->numel;
  int64_t n = std::max(nb, na);
  int64_t m = n - 1;

  if (m == 0) {
    // No delay elements — zi is empty
    return C_SUCCESS;
  }

  const double *bp = (const double *)b_arr->data;
  const double *ap = (const double *)a_arr->data;
  double *zi = (double *)zi_arr->data;

  if (!bp || !ap || !zi) {
    cpu_set_last_error("lfilter_zi: null data pointer");
    return C_FAILED;
  }

  // Build companion matrix A (m x m)
  // A[i][0] = -a[i+1], A[i][i+1] = 1, rest = 0
  // Build (I - A) in augmented form [I-A | B]
  std::vector<double> aug(m * (m + 1), 0.0);
  for (int64_t i = 0; i < m; ++i) {
    double ai1 = (i + 1 < na) ? ap[i + 1] : 0.0;
    // (I - A)[i][j]
    for (int64_t j = 0; j < m; ++j) {
      if (i == j)
        aug[i * (m + 1) + j] = 1.0 + ai1; // 1 - (-a[i+1])
      else if (j == 0)
        aug[i * (m + 1) + j] = ai1; // -(-a[i+1]) for first column
      else if (j == i + 1)
        aug[i * (m + 1) + j] = -1.0; // -(1) for superdiagonal
      // else 0
    }
    // RHS: b[i+1] - a[i+1] * b[0]
    double bi1 = (i + 1 < nb) ? bp[i + 1] : 0.0;
    aug[i * (m + 1) + m] = bi1 - ai1 * bp[0];
  }

  // Gauss-Jordan elimination with partial pivoting
  for (int64_t col = 0; col < m; ++col) {
    // Find pivot
    int64_t pivot_row = col;
    double max_val = std::abs(aug[col * (m + 1) + col]);
    for (int64_t row = col + 1; row < m; ++row) {
      double val = std::abs(aug[row * (m + 1) + col]);
      if (val > max_val) {
        max_val = val;
        pivot_row = row;
      }
    }

    if (max_val < 1e-15) {
      cpu_set_last_error("lfilter_zi: singular matrix");
      return C_FAILED;
    }

    // Swap rows
    if (pivot_row != col) {
      for (int64_t j = 0; j <= m; ++j) {
        std::swap(aug[col * (m + 1) + j], aug[pivot_row * (m + 1) + j]);
      }
    }

    // Scale pivot row
    double pivot = aug[col * (m + 1) + col];
    for (int64_t j = 0; j <= m; ++j) {
      aug[col * (m + 1) + j] /= pivot;
    }

    // Eliminate column
    for (int64_t row = 0; row < m; ++row) {
      if (row == col)
        continue;
      double factor = aug[row * (m + 1) + col];
      for (int64_t j = 0; j <= m; ++j) {
        aug[row * (m + 1) + j] -= factor * aug[col * (m + 1) + j];
      }
    }
  }

  // Extract solution
  for (int64_t i = 0; i < m; ++i) {
    zi[i] = aug[i * (m + 1) + m];
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(lfilter_zi, INSIGHT_DTYPE_F64, lfilter_zi_kernel_cpu);
