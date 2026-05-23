// backends/cpu/kernels/indexing/masked_select.cpp
/**
 * @file masked_select.cpp
 * @brief CPU kernel for masked_select operation.
 * 
 * Selects elements from input array where mask is true.
 * 
 * Input layout:
 *   inputs[0] = InsightArray* output (pre-allocated with correct size)
 *   inputs[1] = InsightArray* input
 *   inputs[2] = InsightArray* mask (bool)
 * 
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status masked_select_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* out = (InsightArray*)outputs[0];
    InsightArray* x = (InsightArray*)inputs[1];
    InsightArray* mask = (InsightArray*)inputs[2];
    
    if (!out || !x || !mask) {
        cpu_set_last_error("masked_select: null array pointer");
        return C_FAILED;
    }
    
    int64_t total = x->numel;
    int64_t out_idx = 0;
    const bool* msk = (const bool*)mask->data;
    
    switch (x->dtype) {
        case INSIGHT_DTYPE_F32: {
            float* dst = (float*)out->data;
            const float* src = (const float*)x->data;
            for (int64_t i = 0; i < total; ++i) {
                if (msk[i]) {
                    dst[out_idx++] = src[i];
                }
            }
            break;
        }
        case INSIGHT_DTYPE_F64: {
            double* dst = (double*)out->data;
            const double* src = (const double*)x->data;
            for (int64_t i = 0; i < total; ++i) {
                if (msk[i]) {
                    dst[out_idx++] = src[i];
                }
            }
            break;
        }
        case INSIGHT_DTYPE_I32: {
            int32_t* dst = (int32_t*)out->data;
            const int32_t* src = (const int32_t*)x->data;
            for (int64_t i = 0; i < total; ++i) {
                if (msk[i]) {
                    dst[out_idx++] = src[i];
                }
            }
            break;
        }
        case INSIGHT_DTYPE_I64: {
            int64_t* dst = (int64_t*)out->data;
            const int64_t* src = (const int64_t*)x->data;
            for (int64_t i = 0; i < total; ++i) {
                if (msk[i]) {
                    dst[out_idx++] = src[i];
                }
            }
            break;
        }
        case INSIGHT_DTYPE_U8: {
            uint8_t* dst = (uint8_t*)out->data;
            const uint8_t* src = (const uint8_t*)x->data;
            for (int64_t i = 0; i < total; ++i) {
                if (msk[i]) {
                    dst[out_idx++] = src[i];
                }
            }
            break;
        }
        case INSIGHT_DTYPE_BOOL: {
            bool* dst = (bool*)out->data;
            const bool* src = (const bool*)x->data;
            for (int64_t i = 0; i < total; ++i) {
                if (msk[i]) {
                    dst[out_idx++] = src[i];
                }
            }
            break;
        }
        default:
            cpu_set_last_error("masked_select: unsupported dtype");
            return C_FAILED;
    }
    
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(masked_select, INSIGHT_DTYPE_F32, masked_select_kernel_cpu);
REGISTER_CPU_KERNEL(masked_select, INSIGHT_DTYPE_F64, masked_select_kernel_cpu);
REGISTER_CPU_KERNEL(masked_select, INSIGHT_DTYPE_I32, masked_select_kernel_cpu);
REGISTER_CPU_KERNEL(masked_select, INSIGHT_DTYPE_I64, masked_select_kernel_cpu);
REGISTER_CPU_KERNEL(masked_select, INSIGHT_DTYPE_U8,  masked_select_kernel_cpu);
REGISTER_CPU_KERNEL(masked_select, INSIGHT_DTYPE_BOOL, masked_select_kernel_cpu);
