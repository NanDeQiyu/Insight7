// backends/cpu/kernels/reduction/min.cpp
/**
 * @file min.cpp
 * @brief CPU kernel for Min reduction.
 * 
 * Performs reduction along the last dimension.
 * Input is guaranteed to be contiguous with shape [batch_size, reduce_size].
 * 
 * @param inputs  [0] = InsightArray* output (shape [batch_size])
 *                [1] = InsightArray* prepared (shape [batch_size, reduce_size])
 *                [2] = int64_t* batch_size
 *                [3] = int64_t* reduce_size
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status min_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* out = (InsightArray*)outputs[0];
    InsightArray* prepared = (InsightArray*)inputs[1];
    
    if (!out || !prepared) {
        cpu_set_last_error("min: null array pointer");
        return C_FAILED;
    }
    
    if (!inputs[2] || !inputs[3]) {
        cpu_set_last_error("min: batch_size or reduce_size is null");
        return C_FAILED;
    }
    
    switch (out->dtype) {
                case INSIGHT_DTYPE_U8:
            REDUCE_MIN_LOOP(uint8_t);
            break;
        case INSIGHT_DTYPE_I8:
            REDUCE_MIN_LOOP(int8_t);
            break;
        case INSIGHT_DTYPE_I16:
            REDUCE_MIN_LOOP(int16_t);
            break;
        case INSIGHT_DTYPE_I32:
            REDUCE_MIN_LOOP(int32_t);
            break;
        case INSIGHT_DTYPE_I64:
            REDUCE_MIN_LOOP(int64_t);
            break;
        case INSIGHT_DTYPE_U16:
            REDUCE_MIN_LOOP(uint16_t);
            break;
        case INSIGHT_DTYPE_U32:
            REDUCE_MIN_LOOP(uint32_t);
            break;
        case INSIGHT_DTYPE_U64:
            REDUCE_MIN_LOOP(uint64_t);
            break;
        case INSIGHT_DTYPE_F32:
            REDUCE_MIN_LOOP(float);
            break;
        case INSIGHT_DTYPE_F64:
            REDUCE_MIN_LOOP(double);
            break;
        default:
            cpu_set_last_error("min: unsupported dtype");
            return C_FAILED;
    }
    
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// Register for all supported types
REGISTER_CPU_KERNEL(min, INSIGHT_DTYPE_U8,   min_kernel_cpu);
REGISTER_CPU_KERNEL(min, INSIGHT_DTYPE_I8,   min_kernel_cpu);
REGISTER_CPU_KERNEL(min, INSIGHT_DTYPE_I16,  min_kernel_cpu);
REGISTER_CPU_KERNEL(min, INSIGHT_DTYPE_I32,  min_kernel_cpu);
REGISTER_CPU_KERNEL(min, INSIGHT_DTYPE_I64,  min_kernel_cpu);
REGISTER_CPU_KERNEL(min, INSIGHT_DTYPE_U16,  min_kernel_cpu);
REGISTER_CPU_KERNEL(min, INSIGHT_DTYPE_U32,  min_kernel_cpu);
REGISTER_CPU_KERNEL(min, INSIGHT_DTYPE_U64,  min_kernel_cpu);
REGISTER_CPU_KERNEL(min, INSIGHT_DTYPE_F32,  min_kernel_cpu);
REGISTER_CPU_KERNEL(min, INSIGHT_DTYPE_F64,  min_kernel_cpu);
