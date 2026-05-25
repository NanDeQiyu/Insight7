// backends/cpu/kernels/unary/conj.cpp
#include "common.h"
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

C_Status conj_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("conj: null array pointer");
    return C_FAILED;
  }

  if (x->numel != out->numel) {
    cpu_set_last_error("conj: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_C32:
    UNARY_KERNEL_LOOP(std::complex<float>,
                      [](std::complex<float> v) { return std::conj(v); });
    break;
  case INSIGHT_DTYPE_C64:
    UNARY_KERNEL_LOOP(std::complex<double>,
                      [](std::complex<double> v) { return std::conj(v); });
    break;
  default:
    // For non-complex, conjugate is identity, but we shouldn't get here
    // because frontend handles identity case
    cpu_set_last_error("conj: only complex types supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(conj, INSIGHT_DTYPE_C32, conj_kernel_cpu);
REGISTER_CPU_KERNEL(conj, INSIGHT_DTYPE_C64, conj_kernel_cpu);
