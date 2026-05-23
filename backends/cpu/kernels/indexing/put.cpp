// backends/cpu/kernels/indexing/put.cpp
/**
 * @file put.cpp
 * @brief CPU kernel for put operation.
 * 
 * Puts values into an array at specified indices.
 * 
 * Input layout:
 *   inputs[0] = InsightArray* output (will be modified)
 *   inputs[1] = InsightArray* indices
 *   inputs[2] = InsightArray* values
 * 
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status put_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* out = (InsightArray*)outputs[0];
    InsightArray* idx = (InsightArray*)inputs[1];
    InsightArray* val = (InsightArray*)inputs[2];
    
    if (!out || !idx || !val) {
        cpu_set_last_error("put: null array pointer");
        return C_FAILED;
    }
    
    int64_t n = idx->numel;
    int64_t out_size = out->numel;
    int64_t val_size = val->numel;
    int64_t* idx_data = (int64_t*)idx->data;
    
    switch (out->dtype) {
        case INSIGHT_DTYPE_F32: {
            float* dst = (float*)out->data;
            const float* v = (const float*)val->data;
            for (int64_t i = 0; i < n; ++i) {
                int64_t pos = idx_data[i];
                if (pos < 0) pos += out_size;
                if (pos < 0 || pos >= out_size) {
                    cpu_set_last_error("put: index out of bounds");
                    return C_FAILED;
                }
                dst[pos] = v[i % val_size];
            }
            break;
        }
        case INSIGHT_DTYPE_F64: {
            double* dst = (double*)out->data;
            const double* v = (const double*)val->data;
            for (int64_t i = 0; i < n; ++i) {
                int64_t pos = idx_data[i];
                if (pos < 0) pos += out_size;
                if (pos < 0 || pos >= out_size) {
                    cpu_set_last_error("put: index out of bounds");
                    return C_FAILED;
                }
                dst[pos] = v[i % val_size];
            }
            break;
        }
        case INSIGHT_DTYPE_I32: {
            int32_t* dst = (int32_t*)out->data;
            const int32_t* v = (const int32_t*)val->data;
            for (int64_t i = 0; i < n; ++i) {
                int64_t pos = idx_data[i];
                if (pos < 0) pos += out_size;
                if (pos < 0 || pos >= out_size) {
                    cpu_set_last_error("put: index out of bounds");
                    return C_FAILED;
                }
                dst[pos] = v[i % val_size];
            }
            break;
        }
        case INSIGHT_DTYPE_I64: {
            int64_t* dst = (int64_t*)out->data;
            const int64_t* v = (const int64_t*)val->data;
            for (int64_t i = 0; i < n; ++i) {
                int64_t pos = idx_data[i];
                if (pos < 0) pos += out_size;
                if (pos < 0 || pos >= out_size) {
                    cpu_set_last_error("put: index out of bounds");
                    return C_FAILED;
                }
                dst[pos] = v[i % val_size];
            }
            break;
        }
        case INSIGHT_DTYPE_U8: {
            uint8_t* dst = (uint8_t*)out->data;
            const uint8_t* v = (const uint8_t*)val->data;
            for (int64_t i = 0; i < n; ++i) {
                int64_t pos = idx_data[i];
                if (pos < 0) pos += out_size;
                if (pos < 0 || pos >= out_size) {
                    cpu_set_last_error("put: index out of bounds");
                    return C_FAILED;
                }
                dst[pos] = v[i % val_size];
            }
            break;
        }
        case INSIGHT_DTYPE_BOOL: {
            bool* dst = (bool*)out->data;
            const bool* v = (const bool*)val->data;
            for (int64_t i = 0; i < n; ++i) {
                int64_t pos = idx_data[i];
                if (pos < 0) pos += out_size;
                if (pos < 0 || pos >= out_size) {
                    cpu_set_last_error("put: index out of bounds");
                    return C_FAILED;
                }
                dst[pos] = v[i % val_size];
            }
            break;
        }
        default:
            cpu_set_last_error("put: unsupported dtype");
            return C_FAILED;
    }
    
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(put, INSIGHT_DTYPE_F32, put_kernel_cpu);
REGISTER_CPU_KERNEL(put, INSIGHT_DTYPE_F64, put_kernel_cpu);
REGISTER_CPU_KERNEL(put, INSIGHT_DTYPE_I32, put_kernel_cpu);
REGISTER_CPU_KERNEL(put, INSIGHT_DTYPE_I64, put_kernel_cpu);
REGISTER_CPU_KERNEL(put, INSIGHT_DTYPE_U8,  put_kernel_cpu);
REGISTER_CPU_KERNEL(put, INSIGHT_DTYPE_BOOL, put_kernel_cpu);
