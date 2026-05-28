// backends/cpu/kernels/reduction/nanquantile.cpp
/**
 * @file nanquantile.cpp
 * @brief CPU kernel for quantile calculation ignoring NaN values.
 *
 * Computes quantile along reduction axis with linear interpolation,
 * skipping NaN values.
 *
 * @param inputs  [0] = InsightArray* output (shape [batch_size])
 *                [1] = InsightArray* prepared (shape [batch_size, reduce_size])
 *                [2] = int64_t* batch_size
 *                [3] = int64_t* reduce_size
 *                [4] = double* q (quantile between 0 and 1)
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <algorithm>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

static void nanquantile_float(const float *src, float *dst, int64_t batch_size,
                              int64_t reduce_size, double q) {

  for (int64_t i = 0; i < batch_size; ++i) {
    std::vector<float> valid;
    valid.reserve(reduce_size);
    for (int64_t j = 0; j < reduce_size; ++j) {
      float val = src[i * reduce_size + j];
      if (!cpu_is_nan_f32(val)) {
        valid.push_back(val);
      }
    }

    if (valid.empty()) {
      dst[i] = 0;
      continue;
    }

    std::sort(valid.begin(), valid.end());
    int64_t n = (int64_t)valid.size();

    if (q == 0.5) {
      if (n % 2 == 1) {
        dst[i] = valid[n / 2];
      } else {
        dst[i] = (valid[n / 2 - 1] + valid[n / 2]) * 0.5f;
      }
    } else {
      double idx = q * (n - 1);
      int64_t lo = (int64_t)idx;
      int64_t hi = lo + 1;
      if (hi >= n) {
        dst[i] = valid[lo];
      } else {
        double frac = idx - lo;
        dst[i] = (float)(valid[lo] * (1 - frac) + valid[hi] * frac);
      }
    }
  }
}

static void nanquantile_double(const double *src, double *dst,
                               int64_t batch_size, int64_t reduce_size,
                               double q) {

  for (int64_t i = 0; i < batch_size; ++i) {
    std::vector<double> valid;
    valid.reserve(reduce_size);
    for (int64_t j = 0; j < reduce_size; ++j) {
      double val = src[i * reduce_size + j];
      if (!cpu_is_nan_f64(val)) {
        valid.push_back(val);
      }
    }

    if (valid.empty()) {
      dst[i] = 0;
      continue;
    }

    std::sort(valid.begin(), valid.end());
    int64_t n = (int64_t)valid.size();

    if (q == 0.5) {
      if (n % 2 == 1) {
        dst[i] = valid[n / 2];
      } else {
        dst[i] = (valid[n / 2 - 1] + valid[n / 2]) * 0.5;
      }
    } else {
      double idx = q * (n - 1);
      int64_t lo = (int64_t)idx;
      int64_t hi = lo + 1;
      if (hi >= n) {
        dst[i] = valid[lo];
      } else {
        double frac = idx - lo;
        dst[i] = valid[lo] * (1 - frac) + valid[hi] * frac;
      }
    }
  }
}

C_Status nanquantile_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *prepared = (InsightArray *)inputs[1];

  if (!out || !prepared) {
    cpu_set_last_error("nanquantile: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2] || !inputs[3] || !inputs[4]) {
    cpu_set_last_error("nanquantile: batch_size, reduce_size, or q is null");
    return C_FAILED;
  }

  int64_t batch_size = *(int64_t *)inputs[2];
  int64_t reduce_size = *(int64_t *)inputs[3];
  double q = *(double *)inputs[4];

  if (q < 0.0 || q > 1.0) {
    cpu_set_last_error("nanquantile: q must be between 0 and 1");
    return C_FAILED;
  }

  switch (prepared->dtype) {
  case INSIGHT_DTYPE_F32:
    nanquantile_float((const float *)prepared->data, (float *)out->data,
                      batch_size, reduce_size, q);
    break;
  case INSIGHT_DTYPE_F64:
    nanquantile_double((const double *)prepared->data, (double *)out->data,
                       batch_size, reduce_size, q);
    break;
  default:
    cpu_set_last_error("nanquantile: only float32 and float64 supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(nanquantile, INSIGHT_DTYPE_F32, nanquantile_kernel_cpu);
REGISTER_CPU_KERNEL(nanquantile, INSIGHT_DTYPE_F64, nanquantile_kernel_cpu);
