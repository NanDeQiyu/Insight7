// backends/cpu/kernels/reduction/nanvar.cpp
/**
 * @file nanvar.cpp
 * @brief CPU kernel for nanvar reduction (variance ignoring NaN).
 * 
 * Computes variance along reduction axis, skipping NaN values.
 * 
 * @param inputs  [0] = InsightArray* output (shape [batch_size])
 *                [1] = InsightArray* prepared (shape [batch_size, reduce_size])
 *                [2] = int64_t* batch_size
 *                [3] = int64_t* reduce_size
 *                [4] = int* ddof (degrees of freedom)
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status nanvar_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* out = (InsightArray*)outputs[0];
    InsightArray* prepared = (InsightArray*)inputs[1];
    
    if (!out || !prepared) {
        cpu_set_last_error("nanvar: null array pointer");
        return C_FAILED;
    }
    
    if (!inputs[2] || !inputs[3] || !inputs[4]) {
        cpu_set_last_error("nanvar: batch_size, reduce_size, or ddof is null");
        return C_FAILED;
    }
    
    switch (prepared->dtype) {
        case INSIGHT_DTYPE_F32:
            REDUCE_NANVAR_LOOP(float, cpu_is_nan_f32);
            break;
        case INSIGHT_DTYPE_F64:
            REDUCE_NANVAR_LOOP(double, cpu_is_nan_f64);
            break;
        default:
            cpu_set_last_error("nanvar: only float32 and float64 supported");
            return C_FAILED;
    }
    
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(nanvar, INSIGHT_DTYPE_F32, nanvar_kernel_cpu);
REGISTER_CPU_KERNEL(nanvar, INSIGHT_DTYPE_F64, nanvar_kernel_cpu);
