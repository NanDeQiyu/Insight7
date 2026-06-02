// backends/cpu/kernels/signal/windows/tukey.cpp
// CPU kernel for Tukey (tapered cosine) window generation.
// If n < alpha*(M-1)/2:
//   w[n] = 0.5*(1 + cos(pi*(2*n/(alpha*(M-1)) - 1)))
// If n > (M-1)*(1 - alpha/2):
//   w[n] = 0.5*(1 + cos(pi*(2*n/(alpha*(M-1)) - 2/alpha + 1)))
// Otherwise: w[n] = 1
// inputs[0]: alpha (1-element F64 array)
// outputs[0]: window output
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status tukey_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *alpha_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!alpha_arr || !out) {
    cpu_set_last_error("tukey: null array pointer");
    return C_FAILED;
  }

  double alpha = *(double *)alpha_arr->data;
  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *data = (double *)out->data;
    if (M == 1) {
      data[0] = 1.0;
      return C_SUCCESS;
    }

    // alpha <= 0 is equivalent to boxcar; alpha >= 1 is equivalent to hann
    if (alpha <= 0.0) {
      for (int64_t i = 0; i < M; ++i)
        data[i] = 1.0;
    } else if (alpha >= 1.0) {
      double scale = 2.0 * M_PI / (M - 1);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
      for (int64_t i = 0; i < M; ++i) {
        data[i] = 0.5 * (1.0 - cos(scale * i));
      }
    } else {
      double width = alpha * (M - 1);
      double half_width = width / 2.0;
      double boundary1 = half_width;
      double boundary2 = (M - 1) - half_width;
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
      for (int64_t i = 0; i < M; ++i) {
        double n = (double)i;
        if (n < boundary1) {
          data[i] = 0.5 * (1.0 + cos(M_PI * (2.0 * n / width - 1.0)));
        } else if (n > boundary2) {
          data[i] =
              0.5 * (1.0 + cos(M_PI * (2.0 * n / width - 2.0 / alpha + 1.0)));
        } else {
          data[i] = 1.0;
        }
      }
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *data = (float *)out->data;
    float alpha_f = (float)alpha;
    if (M == 1) {
      data[0] = 1.0f;
      return C_SUCCESS;
    }

    if (alpha_f <= 0.0f) {
      for (int64_t i = 0; i < M; ++i)
        data[i] = 1.0f;
    } else if (alpha_f >= 1.0f) {
      float scale = 2.0f * (float)M_PI / (M - 1);
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
      for (int64_t i = 0; i < M; ++i) {
        data[i] = 0.5f * (1.0f - cosf(scale * i));
      }
    } else {
      float width = alpha_f * (M - 1);
      float half_width = width / 2.0f;
      float boundary1 = half_width;
      float boundary2 = (M - 1) - half_width;
#ifdef _OPENMP
#pragma omp parallel for if (M > 1000)
#endif
      for (int64_t i = 0; i < M; ++i) {
        float n = (float)i;
        if (n < boundary1) {
          data[i] =
              0.5f * (1.0f + cosf((float)M_PI * (2.0f * n / width - 1.0f)));
        } else if (n > boundary2) {
          data[i] = 0.5f * (1.0f + cosf((float)M_PI * (2.0f * n / width -
                                                       2.0f / alpha_f + 1.0f)));
        } else {
          data[i] = 1.0f;
        }
      }
    }
  } else {
    cpu_set_last_error("tukey: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(tukey, INSIGHT_DTYPE_F64, tukey_kernel_cpu);
REGISTER_CPU_KERNEL(tukey, INSIGHT_DTYPE_F32, tukey_kernel_cpu);
