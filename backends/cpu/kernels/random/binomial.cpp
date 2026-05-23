// backends/cpu/kernels/random/binomial.cpp
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status binomial_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* out = (InsightArray*)outputs[0];
    
    if (!out) {
        cpu_set_last_error("binomial: output array is null");
        return C_FAILED;
    }
    
    int64_t n = *(int64_t*)inputs[1];
    double p = *(double*)inputs[2];
    
    switch (out->dtype) {
        case INSIGHT_DTYPE_I32:
            RANDOM_FILL_LOOP(int32_t, std::binomial_distribution<int32_t>, (n, p));
            break;
        case INSIGHT_DTYPE_I64:
            RANDOM_FILL_LOOP(int64_t, std::binomial_distribution<int64_t>, (n, p));
            break;
        default:
            cpu_set_last_error("binomial: unsupported dtype (need int32/int64)");
            return C_FAILED;
    }
    
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(binomial, INSIGHT_DTYPE_I32, binomial_kernel_cpu);
REGISTER_CPU_KERNEL(binomial, INSIGHT_DTYPE_I64, binomial_kernel_cpu);
