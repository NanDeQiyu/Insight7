// backends/cpu/kernels/reduction/prod.cpp
/**
 * @file prod.cpp
 * @brief CPU kernel for Product reduction.
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

C_Status prod_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* out = (InsightArray*)outputs[0];
    InsightArray* prepared = (InsightArray*)inputs[1];
    
    if (!out || !prepared) {
        cpu_set_last_error("prod: null array pointer");
        return C_FAILED;
    }
    
    if (!inputs[2] || !inputs[3]) {
        cpu_set_last_error("prod: batch_size or reduce_size is null");
        return C_FAILED;
    }
    
    switch (out->dtype) {
                case INSIGHT_DTYPE_U8:
            REDUCE_PROD_LOOP(uint8_t);
            break;
        case INSIGHT_DTYPE_I8:
            REDUCE_PROD_LOOP(int8_t);
            break;
        case INSIGHT_DTYPE_I16:
            REDUCE_PROD_LOOP(int16_t);
            break;
        case INSIGHT_DTYPE_I32:
            REDUCE_PROD_LOOP(int32_t);
            break;
        case INSIGHT_DTYPE_I64:
            REDUCE_PROD_LOOP(int64_t);
            break;
        case INSIGHT_DTYPE_U16:
            REDUCE_PROD_LOOP(uint16_t);
            break;
        case INSIGHT_DTYPE_U32:
            REDUCE_PROD_LOOP(uint32_t);
            break;
        case INSIGHT_DTYPE_U64:
            REDUCE_PROD_LOOP(uint64_t);
            break;
        case INSIGHT_DTYPE_F32:
            REDUCE_PROD_LOOP(float);
            break;
        case INSIGHT_DTYPE_F64:
            REDUCE_PROD_LOOP(double);
            break;
        default:
            cpu_set_last_error("prod: unsupported dtype");
            return C_FAILED;
    }
    
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// Register for all supported types
REGISTER_CPU_KERNEL(prod, INSIGHT_DTYPE_U8,   prod_kernel_cpu);
REGISTER_CPU_KERNEL(prod, INSIGHT_DTYPE_I8,   prod_kernel_cpu);
REGISTER_CPU_KERNEL(prod, INSIGHT_DTYPE_I16,  prod_kernel_cpu);
REGISTER_CPU_KERNEL(prod, INSIGHT_DTYPE_I32,  prod_kernel_cpu);
REGISTER_CPU_KERNEL(prod, INSIGHT_DTYPE_I64,  prod_kernel_cpu);
REGISTER_CPU_KERNEL(prod, INSIGHT_DTYPE_U16,  prod_kernel_cpu);
REGISTER_CPU_KERNEL(prod, INSIGHT_DTYPE_U32,  prod_kernel_cpu);
REGISTER_CPU_KERNEL(prod, INSIGHT_DTYPE_U64,  prod_kernel_cpu);
REGISTER_CPU_KERNEL(prod, INSIGHT_DTYPE_F32,  prod_kernel_cpu);
REGISTER_CPU_KERNEL(prod, INSIGHT_DTYPE_F64,  prod_kernel_cpu);
