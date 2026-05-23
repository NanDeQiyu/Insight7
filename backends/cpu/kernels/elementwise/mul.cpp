// backends/cpu/kernels/elementwise/mul.cpp
/**
 * @file mul.cpp
 * @brief CPU kernel for Multiplication operation.
 * 
 * Computes elementwise (a * b) of two arrays with stride support.
 * Supports all numeric data types including complex numbers.
 * 
 * @param inputs  [0] = InsightArray* left operand
 *                [1] = InsightArray* right operand
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status mul_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* a = (InsightArray*)inputs[0];
    InsightArray* b = (InsightArray*)inputs[1];
    InsightArray* out = (InsightArray*)outputs[0];
    
    if (!a || !b || !out) {
        cpu_set_last_error("mul: null array pointer");
        return C_FAILED;
    }
    
    // Check that shapes are compatible (numel must match)
    if (a->numel != out->numel || b->numel != out->numel) {
        cpu_set_last_error("mul: shape mismatch");
        return C_FAILED;
    }
    
    switch (a->dtype) {
        case INSIGHT_DTYPE_BOOL:
            ELEMENTWISE_KERNEL_LOOP(bool, *);
            break;
        case INSIGHT_DTYPE_U8:
            ELEMENTWISE_KERNEL_LOOP(uint8_t, *);
            break;
        case INSIGHT_DTYPE_I8:
            ELEMENTWISE_KERNEL_LOOP(int8_t, *);
            break;
        case INSIGHT_DTYPE_I16:
            ELEMENTWISE_KERNEL_LOOP(int16_t, *);
            break;
        case INSIGHT_DTYPE_I32:
            ELEMENTWISE_KERNEL_LOOP(int32_t, *);
            break;
        case INSIGHT_DTYPE_I64:
            ELEMENTWISE_KERNEL_LOOP(int64_t, *);
            break;
        case INSIGHT_DTYPE_U16:
            ELEMENTWISE_KERNEL_LOOP(uint16_t, *);
            break;
        case INSIGHT_DTYPE_U32:
            ELEMENTWISE_KERNEL_LOOP(uint32_t, *);
            break;
        case INSIGHT_DTYPE_U64:
            ELEMENTWISE_KERNEL_LOOP(uint64_t, *);
            break;
        case INSIGHT_DTYPE_F32:
            ELEMENTWISE_KERNEL_LOOP(float, *);
            break;
        case INSIGHT_DTYPE_F64:
            ELEMENTWISE_KERNEL_LOOP(double, *);
            break;
        case INSIGHT_DTYPE_C32:
            ELEMENTWISE_KERNEL_LOOP(std::complex<float>, *);
            break;
        case INSIGHT_DTYPE_C64:
            ELEMENTWISE_KERNEL_LOOP(std::complex<double>, *);
            break;
        default:
            cpu_set_last_error("mul: unsupported dtype");
            return C_FAILED;
    }
    
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

// Register for all supported types
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_BOOL, mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_U8,   mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_I8,   mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_I16,  mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_I32,  mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_I64,  mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_U16,  mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_U32,  mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_U64,  mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_F16,  mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_BF16, mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_F32,  mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_F64,  mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_C32,  mul_kernel_cpu);
REGISTER_CPU_KERNEL(mul, INSIGHT_DTYPE_C64,  mul_kernel_cpu);
