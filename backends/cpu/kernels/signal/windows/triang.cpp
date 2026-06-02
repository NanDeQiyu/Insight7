// backends/cpu/kernels/signal/windows/triang.cpp
// CPU kernel for triangular window generation.
// Matches scipy.signal.windows.triang implementation.
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status triang_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];

  if (!out) {
    cpu_set_last_error("triang: null output pointer");
    return C_FAILED;
  }

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    if (M == 1) {
      data[0] = 1.0;
      return C_SUCCESS;
    }

    if (M % 2 == 1) {
      // Odd M: w[n] = (n+1)/(M/2+1) for n <= (M-1)/2
      //          w[n] = (M-n)/(M/2+1) for n >= (M-1)/2
      double denom = (double)(M / 2 + 1);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
      for (int64_t n = 0; n < M; ++n) {
        if (n <= (M - 1) / 2) {
          data[n] = (n + 1) / denom;
        } else {
          data[n] = (M - n) / denom;
        }
      }
    } else {
      // Even M: w[n] = (n+1)/(M/2) for n < M/2
      //          w[n] = (M-n)/(M/2) for n >= M/2
      double denom = (double)(M / 2);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
      for (int64_t n = 0; n < M; ++n) {
        if (n < M / 2) {
          data[n] = (n + 1) / denom;
        } else {
          data[n] = (M - n) / denom;
        }
      }
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    if (M == 1) {
      data[0] = 1.0f;
      return C_SUCCESS;
    }

    if (M % 2 == 1) {
      float denom = (float)(M / 2 + 1);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
      for (int64_t n = 0; n < M; ++n) {
        if (n <= (M - 1) / 2) {
          data[n] = (n + 1) / denom;
        } else {
          data[n] = (M - n) / denom;
        }
      }
    } else {
      float denom = (float)(M / 2);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
      for (int64_t n = 0; n < M; ++n) {
        if (n < M / 2) {
          data[n] = (n + 1) / denom;
        } else {
          data[n] = (M - n) / denom;
        }
      }
    }
  } else {
    cpu_set_last_error("triang: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(triang, INSIGHT_DTYPE_F64, triang_kernel_cpu);
REGISTER_CPU_KERNEL(triang, INSIGHT_DTYPE_F32, triang_kernel_cpu);
