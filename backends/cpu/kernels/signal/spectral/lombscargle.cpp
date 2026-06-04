// backends/cpu/kernels/signal/spectral/lombscargle.cpp
// CPU kernel for Lomb-Scargle periodogram.
// For each frequency f, compute:
//   tau = atan2(sum(sin(2*w*t)), sum(cos(2*w*t))) / (2*w)
//   P[f] = 0.5 * ( (sum(y*cos(w*(t-tau)))^2 / sum(cos^2(w*(t-tau)))
//                 + (sum(y*sin(w*(t-tau)))^2 / sum(sin^2(w*(t-tau))) )
// inputs[0]: x (sample times, F64)
// inputs[1]: y (sample values, F64)
// inputs[2]: freqs (evaluation frequencies, F64)
// outputs[0]: power spectral density (F64)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

C_Status signal_lombscargle_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *y_arr = (InsightArray *)inputs[1];
  InsightArray *freqs_arr = (InsightArray *)inputs[2];
  InsightArray *p_arr = (InsightArray *)outputs[0];

  if (!x_arr || !y_arr || !freqs_arr || !p_arr) {
    cpu_set_last_error("signal_lombscargle: null array pointer");
    return C_FAILED;
  }

  if (!x_arr->data || !y_arr->data || !freqs_arr->data || !p_arr->data) {
    cpu_set_last_error("signal_lombscargle: null data pointer");
    return C_FAILED;
  }

  int64_t n = x_arr->numel;
  int64_t nf = freqs_arr->numel;

  switch (x_arr->dtype) {
  case INSIGHT_DTYPE_F64: {
    const double *x = (const double *)x_arr->data;
    const double *y = (const double *)y_arr->data;
    const double *freqs = (const double *)freqs_arr->data;
    double *p = (double *)p_arr->data;

#ifdef _OPENMP
    if (nf > 100) {
#pragma omp parallel for schedule(dynamic)
      for (int64_t fi = 0; fi < nf; ++fi) {
        double f = freqs[fi];
        double w = 2.0 * M_PI * f;

        if (w == 0.0) {
          double sum_y = 0.0;
          for (int64_t j = 0; j < n; ++j)
            sum_y += y[j];
          p[fi] = (sum_y * sum_y) / (double)(n * n);
          continue;
        }

        double sin2wt = 0.0, cos2wt = 0.0;
        for (int64_t j = 0; j < n; ++j) {
          double arg = 2.0 * w * x[j];
          sin2wt += std::sin(arg);
          cos2wt += std::cos(arg);
        }
        double tau = std::atan2(sin2wt, cos2wt) / (2.0 * w);

        double cs = 0.0, cc = 0.0, sn = 0.0, ss = 0.0;
        for (int64_t j = 0; j < n; ++j) {
          double arg = w * (x[j] - tau);
          double c = std::cos(arg);
          double s = std::sin(arg);
          cs += y[j] * c;
          cc += c * c;
          sn += y[j] * s;
          ss += s * s;
        }

        double pval = 0.0;
        if (cc > 1e-30)
          pval += (cs * cs) / cc;
        if (ss > 1e-30)
          pval += (sn * sn) / ss;
        p[fi] = 0.5 * pval;
      }
    } else {
#endif
      for (int64_t fi = 0; fi < nf; ++fi) {
        double f = freqs[fi];
        double w = 2.0 * M_PI * f;

        if (w == 0.0) {
          double sum_y = 0.0;
          for (int64_t j = 0; j < n; ++j)
            sum_y += y[j];
          p[fi] = (sum_y * sum_y) / (double)(n * n);
          continue;
        }

        double sin2wt = 0.0, cos2wt = 0.0;
        for (int64_t j = 0; j < n; ++j) {
          double arg = 2.0 * w * x[j];
          sin2wt += std::sin(arg);
          cos2wt += std::cos(arg);
        }
        double tau = std::atan2(sin2wt, cos2wt) / (2.0 * w);

        double cs = 0.0, cc = 0.0, sn = 0.0, ss = 0.0;
        for (int64_t j = 0; j < n; ++j) {
          double arg = w * (x[j] - tau);
          double c = std::cos(arg);
          double s = std::sin(arg);
          cs += y[j] * c;
          cc += c * c;
          sn += y[j] * s;
          ss += s * s;
        }

        double pval = 0.0;
        if (cc > 1e-30)
          pval += (cs * cs) / cc;
        if (ss > 1e-30)
          pval += (sn * sn) / ss;
        p[fi] = 0.5 * pval;
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  case INSIGHT_DTYPE_F32: {
    const float *x = (const float *)x_arr->data;
    const float *y = (const float *)y_arr->data;
    const float *freqs = (const float *)freqs_arr->data;
    float *p = (float *)p_arr->data;

#ifdef _OPENMP
    if (nf > 100) {
#pragma omp parallel for schedule(dynamic)
      for (int64_t fi = 0; fi < nf; ++fi) {
        float f = freqs[fi];
        float w = 2.0f * (float)M_PI * f;

        if (w == 0.0f) {
          float sum_y = 0.0f;
          for (int64_t j = 0; j < n; ++j)
            sum_y += y[j];
          p[fi] = (sum_y * sum_y) / (float)(n * n);
          continue;
        }

        float sin2wt = 0.0f, cos2wt = 0.0f;
        for (int64_t j = 0; j < n; ++j) {
          float arg = 2.0f * w * x[j];
          sin2wt += std::sin(arg);
          cos2wt += std::cos(arg);
        }
        float tau = std::atan2(sin2wt, cos2wt) / (2.0f * w);

        float cs = 0.0f, cc = 0.0f, sn = 0.0f, ss = 0.0f;
        for (int64_t j = 0; j < n; ++j) {
          float arg = w * (x[j] - tau);
          float c = std::cos(arg);
          float s = std::sin(arg);
          cs += y[j] * c;
          cc += c * c;
          sn += y[j] * s;
          ss += s * s;
        }

        float pval = 0.0f;
        if (cc > 1e-30f)
          pval += (cs * cs) / cc;
        if (ss > 1e-30f)
          pval += (sn * sn) / ss;
        p[fi] = 0.5f * pval;
      }
    } else {
#endif
      for (int64_t fi = 0; fi < nf; ++fi) {
        float f = freqs[fi];
        float w = 2.0f * (float)M_PI * f;

        if (w == 0.0f) {
          float sum_y = 0.0f;
          for (int64_t j = 0; j < n; ++j)
            sum_y += y[j];
          p[fi] = (sum_y * sum_y) / (float)(n * n);
          continue;
        }

        float sin2wt = 0.0f, cos2wt = 0.0f;
        for (int64_t j = 0; j < n; ++j) {
          float arg = 2.0f * w * x[j];
          sin2wt += std::sin(arg);
          cos2wt += std::cos(arg);
        }
        float tau = std::atan2(sin2wt, cos2wt) / (2.0f * w);

        float cs = 0.0f, cc = 0.0f, sn = 0.0f, ss = 0.0f;
        for (int64_t j = 0; j < n; ++j) {
          float arg = w * (x[j] - tau);
          float c = std::cos(arg);
          float s = std::sin(arg);
          cs += y[j] * c;
          cc += c * c;
          sn += y[j] * s;
          ss += s * s;
        }

        float pval = 0.0f;
        if (cc > 1e-30f)
          pval += (cs * cs) / cc;
        if (ss > 1e-30f)
          pval += (sn * sn) / ss;
        p[fi] = 0.5f * pval;
      }
#ifdef _OPENMP
    }
#endif
    break;
  }
  default:
    cpu_set_last_error(
        "signal_lombscargle: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_CPU_KERNEL(signal_lombscargle, INSIGHT_DTYPE_F64,
                    signal_lombscargle_kernel_cpu);
REGISTER_CPU_KERNEL(signal_lombscargle, INSIGHT_DTYPE_F32,
                    signal_lombscargle_kernel_cpu);
