// backends/cpu/kernels/unary/exp2.cpp
#include "common.h"
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

C_Status exp2_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* x = (InsightArray*)inputs[0];
    InsightArray* out = (InsightArray*)outputs[0];
    
    if (!x || !out) {
        cpu_set_last_error("exp2: null array pointer");
        return C_FAILED;
    }
    
    if (x->numel != out->numel) {
        cpu_set_last_error("exp2: shape mismatch");
        return C_FAILED;
    }
    
    switch (x->dtype) {
        case INSIGHT_DTYPE_F32:
            UNARY_KERNEL_LOOP(float, [](float v) { return std::exp2(v); });
            break;
        case INSIGHT_DTYPE_F64:
            UNARY_KERNEL_LOOP(double, [](double v) { return std::exp2(v); });
            break;
        default:
            cpu_set_last_error("exp2: only float32 and float64 supported");
            return C_FAILED;
    }
    
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(exp2, INSIGHT_DTYPE_F32, exp2_kernel_cpu);
REGISTER_CPU_KERNEL(exp2, INSIGHT_DTYPE_F64, exp2_kernel_cpu);
