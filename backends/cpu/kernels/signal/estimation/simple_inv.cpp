// backends/cpu/kernels/signal/estimation/simple_inv.cpp
// Gauss-Jordan elimination for matrix inverse (no LAPACK dependency)
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status simple_inv_kernel_cpu(void **inputs, void **outputs) {
  // inputs[0] = matrix (N×N)
  // outputs[0] = inverse (N×N)
  InsightArray *in = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!in || !out) {
    cpu_set_last_error("simple_inv: null array pointer");
    return C_FAILED;
  }

  if (in->ndim != 2 || in->dims[0] != in->dims[1]) {
    cpu_set_last_error("simple_inv: input must be a square 2D matrix");
    return C_FAILED;
  }

  int64_t n = in->dims[0];

  switch (in->dtype) {
  case INSIGHT_DTYPE_F64: {
    const double *src = (const double *)in->data;
    double *dst = (double *)out->data;

    // Build augmented matrix [A | I] in dst (column-major: col * n + row)
    // First copy input to working buffer
    double *A = new double[2 * n * n];
    for (int64_t i = 0; i < n * n; ++i)
      A[i] = src[i];
    // Identity in right half
    for (int64_t i = 0; i < n * n; ++i)
      A[n * n + i] = 0.0;
    for (int64_t i = 0; i < n; ++i)
      A[n * n + i * n + i] = 1.0;

    // Forward elimination with partial pivoting
    for (int64_t col = 0; col < n; ++col) {
      // Find pivot
      int64_t pivot = col;
      double max_val = std::abs(A[col * n + col]);
      for (int64_t row = col + 1; row < n; ++row) {
        double v = std::abs(A[col * n + row]);
        if (v > max_val) {
          max_val = v;
          pivot = row;
        }
      }

      if (max_val < 1e-15) {
        delete[] A;
        cpu_set_last_error("simple_inv: singular matrix");
        return C_FAILED;
      }

      // Swap rows if needed
      if (pivot != col) {
        for (int64_t j = 0; j < 2 * n; ++j) {
          double tmp = A[j * n + col];
          A[j * n + col] = A[j * n + pivot];
          A[j * n + pivot] = tmp;
        }
      }

      // Scale pivot row
      double piv = A[col * n + col];
      for (int64_t j = 0; j < 2 * n; ++j) {
        A[j * n + col] /= piv;
      }

      // Eliminate column
      for (int64_t row = 0; row < n; ++row) {
        if (row == col)
          continue;
        double factor = A[col * n + row];
        for (int64_t j = 0; j < 2 * n; ++j) {
          A[j * n + row] -= factor * A[j * n + col];
        }
      }
    }

    // Copy right half to output
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        dst[j * n + i] = A[(n + j) * n + i];
      }
    }

    delete[] A;
    break;
  }
  case INSIGHT_DTYPE_F32: {
    const float *src = (const float *)in->data;
    float *dst = (float *)out->data;

    float *A = new float[2 * n * n];
    for (int64_t i = 0; i < n * n; ++i)
      A[i] = src[i];
    for (int64_t i = 0; i < n * n; ++i)
      A[n * n + i] = 0.0f;
    for (int64_t i = 0; i < n; ++i)
      A[n * n + i * n + i] = 1.0f;

    for (int64_t col = 0; col < n; ++col) {
      int64_t pivot = col;
      float max_val = std::abs(A[col * n + col]);
      for (int64_t row = col + 1; row < n; ++row) {
        float v = std::abs(A[col * n + row]);
        if (v > max_val) {
          max_val = v;
          pivot = row;
        }
      }

      if (max_val < 1e-7f) {
        delete[] A;
        cpu_set_last_error("simple_inv: singular matrix");
        return C_FAILED;
      }

      if (pivot != col) {
        for (int64_t j = 0; j < 2 * n; ++j) {
          float tmp = A[j * n + col];
          A[j * n + col] = A[j * n + pivot];
          A[j * n + pivot] = tmp;
        }
      }

      float piv = A[col * n + col];
      for (int64_t j = 0; j < 2 * n; ++j) {
        A[j * n + col] /= piv;
      }

      for (int64_t row = 0; row < n; ++row) {
        if (row == col)
          continue;
        float factor = A[col * n + row];
        for (int64_t j = 0; j < 2 * n; ++j) {
          A[j * n + row] -= factor * A[j * n + col];
        }
      }
    }

    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        dst[j * n + i] = A[(n + j) * n + i];
      }
    }

    delete[] A;
    break;
  }
  default:
    cpu_set_last_error("simple_inv: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(simple_inv, INSIGHT_DTYPE_F32, simple_inv_kernel_cpu);
REGISTER_CPU_KERNEL(simple_inv, INSIGHT_DTYPE_F64, simple_inv_kernel_cpu);
