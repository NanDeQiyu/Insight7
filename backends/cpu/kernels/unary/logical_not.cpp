// backends/cpu/kernels/unary/logical_not.cpp
#include "common.h"
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

C_Status logical_not_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* x = (InsightArray*)inputs[0];
    InsightArray* out = (InsightArray*)outputs[0];
    
    if (!x || !out) {
        cpu_set_last_error("logical_not: null array pointer");
        return C_FAILED;
    }
    
    if (x->numel != out->numel) {
        cpu_set_last_error("logical_not: shape mismatch");
        return C_FAILED;
    }
    
    switch (x->dtype) {
        case INSIGHT_DTYPE_BOOL:
            UNARY_CMP_LOOP(bool, [](bool v) { return !v; });
            break;
        case INSIGHT_DTYPE_U8:
            UNARY_CMP_LOOP(uint8_t, [](uint8_t v) { return v == 0; });
            break;
        case INSIGHT_DTYPE_I8:
            UNARY_CMP_LOOP(int8_t, [](int8_t v) { return v == 0; });
            break;
        case INSIGHT_DTYPE_I16:
            UNARY_CMP_LOOP(int16_t, [](int16_t v) { return v == 0; });
            break;
        case INSIGHT_DTYPE_I32:
            UNARY_CMP_LOOP(int32_t, [](int32_t v) { return v == 0; });
            break;
        case INSIGHT_DTYPE_I64:
            UNARY_CMP_LOOP(int64_t, [](int64_t v) { return v == 0; });
            break;
        case INSIGHT_DTYPE_U16:
            UNARY_CMP_LOOP(uint16_t, [](uint16_t v) { return v == 0; });
            break;
        case INSIGHT_DTYPE_U32:
            UNARY_CMP_LOOP(uint32_t, [](uint32_t v) { return v == 0; });
            break;
        case INSIGHT_DTYPE_U64:
            UNARY_CMP_LOOP(uint64_t, [](uint64_t v) { return v == 0; });
            break;
        case INSIGHT_DTYPE_F32:
            UNARY_CMP_LOOP(float, [](float v) { return v == 0.0f; });
            break;
        case INSIGHT_DTYPE_F64:
            UNARY_CMP_LOOP(double, [](double v) { return v == 0.0; });
            break;
        case INSIGHT_DTYPE_C32:
            UNARY_CMP_LOOP(std::complex<float>, [](std::complex<float> v) { return v == std::complex<float>(0, 0); });
            break;
        case INSIGHT_DTYPE_C64:
            UNARY_CMP_LOOP(std::complex<double>, [](std::complex<double> v) { return v == std::complex<double>(0, 0); });
            break;
        default:
            cpu_set_last_error("logical_not: unsupported dtype");
            return C_FAILED;
    }
    
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_BOOL, logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_U8,   logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_I8,   logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_I16,  logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_I32,  logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_I64,  logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_U16,  logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_U32,  logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_U64,  logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_F32,  logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_F64,  logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_C32,  logical_not_kernel_cpu);
REGISTER_CPU_KERNEL(logical_not, INSIGHT_DTYPE_C64,  logical_not_kernel_cpu);
