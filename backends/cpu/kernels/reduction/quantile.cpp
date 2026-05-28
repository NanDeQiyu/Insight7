// backends/cpu/kernels/reduction/quantile.cpp
/**
 * @file quantile.cpp
 * @brief CPU kernel for quantile calculation.
 *
 * Computes quantile along reduction axis with linear interpolation.
 *
 * @param inputs  [0] = InsightArray* output (shape [batch_size], dtype float64)
 *                [1] = InsightArray* prepared (shape [batch_size, reduce_size])
 *                [2] = int64_t* batch_size
 *                [3] = int64_t* reduce_size
 *                [4] = double* q (quantile between 0 and 1)
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <algorithm>

#ifdef __cplusplus
extern "C" {
#endif

static void quantile_float(const float *src, double *dst, int64_t batch_size,
                           int64_t reduce_size, double q) {

  std::vector<float> row(reduce_size);

  for (int64_t i = 0; i < batch_size; ++i) {
    for (int64_t j = 0; j < reduce_size; ++j) {
      row[j] = src[i * reduce_size + j];
    }
    std::sort(row.begin(), row.end());

    if (q == 0.5) {
      if (reduce_size % 2 == 1) {
        dst[i] = row[reduce_size / 2];
      } else {
        dst[i] = (row[reduce_size / 2 - 1] + row[reduce_size / 2]) * 0.5;
      }
    } else {
      double idx = q * (reduce_size - 1);
      int64_t lo = (int64_t)idx;
      int64_t hi = lo + 1;
      if (hi >= reduce_size) {
        dst[i] = row[lo];
      } else {
        double frac = idx - lo;
        dst[i] = row[lo] * (1 - frac) + row[hi] * frac;
      }
    }
  }
}

static void quantile_double(const double *src, double *dst, int64_t batch_size,
                            int64_t reduce_size, double q) {

  std::vector<double> row(reduce_size);

  for (int64_t i = 0; i < batch_size; ++i) {
    for (int64_t j = 0; j < reduce_size; ++j) {
      row[j] = src[i * reduce_size + j];
    }
    std::sort(row.begin(), row.end());

    if (q == 0.5) {
      if (reduce_size % 2 == 1) {
        dst[i] = row[reduce_size / 2];
      } else {
        dst[i] = (row[reduce_size / 2 - 1] + row[reduce_size / 2]) * 0.5;
      }
    } else {
      double idx = q * (reduce_size - 1);
      int64_t lo = (int64_t)idx;
      int64_t hi = lo + 1;
      if (hi >= reduce_size) {
        dst[i] = row[lo];
      } else {
        double frac = idx - lo;
        dst[i] = row[lo] * (1 - frac) + row[hi] * frac;
      }
    }
  }
}

C_Status quantile_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *prepared = (InsightArray *)inputs[1];

  if (!out || !prepared) {
    cpu_set_last_error("quantile: null array pointer");
    return C_FAILED;
  }

  if (!inputs[2] || !inputs[3] || !inputs[4]) {
    cpu_set_last_error("quantile: batch_size, reduce_size, or q is null");
    return C_FAILED;
  }

  int64_t batch_size = *(int64_t *)inputs[2];
  int64_t reduce_size = *(int64_t *)inputs[3];
  double q = *(double *)inputs[4];

  if (q < 0.0 || q > 1.0) {
    cpu_set_last_error("quantile: q must be between 0 and 1");
    return C_FAILED;
  }

  switch (prepared->dtype) {
  case INSIGHT_DTYPE_F32:
    quantile_float((const float *)prepared->data, (double *)out->data,
                   batch_size, reduce_size, q);
    break;
  case INSIGHT_DTYPE_F64:
    quantile_double((const double *)prepared->data, (double *)out->data,
                    batch_size, reduce_size, q);
    break;
  default:
    cpu_set_last_error("quantile: only float32 and float64 supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(quantile, INSIGHT_DTYPE_F32, quantile_kernel_cpu);
REGISTER_CPU_KERNEL(quantile, INSIGHT_DTYPE_F64, quantile_kernel_cpu);
