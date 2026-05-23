// backends/cpu/kernels/unary/abs.cpp
#include "common.h"
#include <cmath>
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

C_Status abs_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* x = (InsightArray*)inputs[0];
    InsightArray* out = (InsightArray*)outputs[0];
    
    if (!x || !out) {
        cpu_set_last_error("abs: null array pointer");
        return C_FAILED;
    }
    
    if (x->numel != out->numel) {
        cpu_set_last_error("abs: shape mismatch");
        return C_FAILED;
    }
    
    switch (x->dtype) {
        case INSIGHT_DTYPE_U8:
            UNARY_KERNEL_LOOP(uint8_t, [](uint8_t v) { return v; });
            break;
        case INSIGHT_DTYPE_I8:
            UNARY_KERNEL_LOOP(int8_t, [](int8_t v) { return v < 0 ? -v : v; });
            break;
        case INSIGHT_DTYPE_I16:
            UNARY_KERNEL_LOOP(int16_t, [](int16_t v) { return v < 0 ? -v : v; });
            break;
        case INSIGHT_DTYPE_I32:
            UNARY_KERNEL_LOOP(int32_t, [](int32_t v) { return v < 0 ? -v : v; });
            break;
        case INSIGHT_DTYPE_I64:
            UNARY_KERNEL_LOOP(int64_t, [](int64_t v) { return v < 0 ? -v : v; });
            break;
        case INSIGHT_DTYPE_U16:
            UNARY_KERNEL_LOOP(uint16_t, [](uint16_t v) { return v; });
            break;
        case INSIGHT_DTYPE_U32:
            UNARY_KERNEL_LOOP(uint32_t, [](uint32_t v) { return v; });
            break;
        case INSIGHT_DTYPE_U64:
            UNARY_KERNEL_LOOP(uint64_t, [](uint64_t v) { return v; });
            break;
        case INSIGHT_DTYPE_F32:
            UNARY_KERNEL_LOOP(float, [](float v) { return std::fabs(v); });
            break;
        case INSIGHT_DTYPE_F64:
            UNARY_KERNEL_LOOP(double, [](double v) { return std::fabs(v); });
            break;
        case INSIGHT_DTYPE_C32:
            UNARY_KERNEL_LOOP(std::complex<float>, [](std::complex<float> v) { return std::abs(v); });
            break;
        case INSIGHT_DTYPE_C64:
            UNARY_KERNEL_LOOP(std::complex<double>, [](std::complex<double> v) { return std::abs(v); });
            break;
        default:
            cpu_set_last_error("abs: unsupported dtype");
            return C_FAILED;
    }
    
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_U8,  abs_kernel_cpu);
REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_I8,  abs_kernel_cpu);
REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_I16, abs_kernel_cpu);
REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_I32, abs_kernel_cpu);
REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_I64, abs_kernel_cpu);
REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_U16, abs_kernel_cpu);
REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_U32, abs_kernel_cpu);
REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_U64, abs_kernel_cpu);
REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_F32, abs_kernel_cpu);
REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_F64, abs_kernel_cpu);
REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_C32, abs_kernel_cpu);
REGISTER_CPU_KERNEL(abs, INSIGHT_DTYPE_C64, abs_kernel_cpu);
