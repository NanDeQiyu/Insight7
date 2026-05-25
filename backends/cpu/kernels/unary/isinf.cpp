// backends/cpu/kernels/unary/isinf.cpp
/**
 * @file isinf.cpp
 * @brief CPU kernel for isinf operation.
 */

#include "common.h"
#include <cmath>
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

C_Status isinf_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("isinf: null array pointer");
    return C_FAILED;
  }

  if (x->numel != out->numel) {
    cpu_set_last_error("isinf: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_BOOL:
  case INSIGHT_DTYPE_U8:
  case INSIGHT_DTYPE_I8:
  case INSIGHT_DTYPE_I16:
  case INSIGHT_DTYPE_I32:
  case INSIGHT_DTYPE_I64:
  case INSIGHT_DTYPE_U16:
  case INSIGHT_DTYPE_U32:
  case INSIGHT_DTYPE_U64:
  case INSIGHT_DTYPE_F16:
  case INSIGHT_DTYPE_BF16:
    // Integer and half types: never infinite
    UNARY_CMP_LOOP(uint8_t, [](uint8_t v) { return false; });
    break;
  case INSIGHT_DTYPE_F32:
    UNARY_CMP_LOOP(float, [](float v) { return std::isinf(v); });
    break;
  case INSIGHT_DTYPE_F64:
    UNARY_CMP_LOOP(double, [](double v) { return std::isinf(v); });
    break;
  case INSIGHT_DTYPE_C32:
    UNARY_CMP_LOOP(std::complex<float>, [](std::complex<float> v) {
      return std::isinf(v.real()) || std::isinf(v.imag());
    });
    break;
  case INSIGHT_DTYPE_C64:
    UNARY_CMP_LOOP(std::complex<double>, [](std::complex<double> v) {
      return std::isinf(v.real()) || std::isinf(v.imag());
    });
    break;
  default:
    cpu_set_last_error("isinf: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_BOOL, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_U8, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_I8, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_I16, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_I32, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_I64, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_U16, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_U32, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_U64, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_F16, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_BF16, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_F32, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_F64, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_C32, isinf_kernel_cpu);
REGISTER_CPU_KERNEL(isinf, INSIGHT_DTYPE_C64, isinf_kernel_cpu);