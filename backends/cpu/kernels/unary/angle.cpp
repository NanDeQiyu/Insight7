// backends/cpu/kernels/unary/angle.cpp
#include "common.h"
#include <cmath>
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

C_Status angle_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("angle: null array pointer");
    return C_FAILED;
  }

  if (x->numel != out->numel) {
    cpu_set_last_error("angle: shape mismatch");
    return C_FAILED;
  }

  int64_t ndim = out->ndim;
  int64_t dims[INSIGHT_MAX_NDIM];
  int64_t x_strides[INSIGHT_MAX_NDIM];
  int64_t out_strides[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    dims[i] = out->dims[i];
    x_strides[i] = x->strides[i];
    out_strides[i] = out->strides[i];
  }
  int64_t n = out->numel;

  switch (x->dtype) {
  case INSIGHT_DTYPE_C32: {
    const std::complex<float> *x_data = (const std::complex<float> *)x->data;
    float *out_data = (float *)out->data;
    _Pragma("omp parallel for") for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_x = cpu_offset_from_linear(linear, ndim, dims, x_strides);
      int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
      std::complex<float> z = x_data[off_x];
      out_data[off_out] = std::atan2(z.imag(), z.real());
    }
    break;
  }
  case INSIGHT_DTYPE_C64: {
    const std::complex<double> *x_data = (const std::complex<double> *)x->data;
    double *out_data = (double *)out->data;
    _Pragma("omp parallel for") for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_x = cpu_offset_from_linear(linear, ndim, dims, x_strides);
      int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
      std::complex<double> z = x_data[off_x];
      out_data[off_out] = std::atan2(z.imag(), z.real());
    }
    break;
  }
  case INSIGHT_DTYPE_F32: {
    const float *x_data = (const float *)x->data;
    float *out_data = (float *)out->data;
    _Pragma("omp parallel for") for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_x = cpu_offset_from_linear(linear, ndim, dims, x_strides);
      int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = std::atan2(0.0f, x_data[off_x]);
    }
    break;
  }
  case INSIGHT_DTYPE_F64: {
    const double *x_data = (const double *)x->data;
    double *out_data = (double *)out->data;
    _Pragma("omp parallel for") for (int64_t linear = 0; linear < n; ++linear) {
      int64_t off_x = cpu_offset_from_linear(linear, ndim, dims, x_strides);
      int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
      out_data[off_out] = std::atan2(0.0, x_data[off_x]);
    }
    break;
  }
  default:
    cpu_set_last_error("angle: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(angle, INSIGHT_DTYPE_C32, angle_kernel_cpu);
REGISTER_CPU_KERNEL(angle, INSIGHT_DTYPE_C64, angle_kernel_cpu);
REGISTER_CPU_KERNEL(angle, INSIGHT_DTYPE_F32, angle_kernel_cpu);
REGISTER_CPU_KERNEL(angle, INSIGHT_DTYPE_F64, angle_kernel_cpu);
