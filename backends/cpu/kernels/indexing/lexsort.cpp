// backends/cpu/kernels/indexing/lexsort.cpp
/**
 * @file lexsort.cpp
 * @brief CPU kernel for lexsort operation.
 * 
 * Performs indirect stable sort using a sequence of keys.
 * 
 * Input layout:
 *   inputs[0] = InsightArray* output (indices)
 *   inputs[1] = InsightArray* transposed (keys flattened)
 *   inputs[2] = int64_t* batch_size
 *   inputs[3] = int64_t* last_dim
 *   inputs[4] = int64_t* nkeys
 * 
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <algorithm>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

C_Status lexsort_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* out = (InsightArray*)outputs[0];
    InsightArray* transposed = (InsightArray*)inputs[1];
    int64_t batch_size = *(int64_t*)inputs[2];
    int64_t last_dim = *(int64_t*)inputs[3];
    int64_t nkeys = *(int64_t*)inputs[4];
    
    if (!out || !transposed) {
        cpu_set_last_error("lexsort: null array pointer");
        return C_FAILED;
    }
    
    int64_t* dst = (int64_t*)out->data;
    
    switch (transposed->dtype) {
        case INSIGHT_DTYPE_F32: {
            const float* src = (const float*)transposed->data;
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                std::vector<std::vector<float>> batch_keys(nkeys);
                for (int64_t k = 0; k < nkeys; ++k) {
                    batch_keys[k].resize(last_dim);
                    const float* key_start = src + (batch * nkeys + k) * last_dim;
                    for (int64_t i = 0; i < last_dim; ++i) {
                        batch_keys[k][i] = key_start[i];
                    }
                }
                
                std::vector<int64_t> indices(last_dim);
                for (int64_t i = 0; i < last_dim; ++i) indices[i] = i;
                
                std::sort(indices.begin(), indices.end(),
                    [&](int64_t a, int64_t b) {
                        for (int64_t k = 0; k < nkeys; ++k) {
                            if (batch_keys[k][a] < batch_keys[k][b]) return true;
                            if (batch_keys[k][a] > batch_keys[k][b]) return false;
                        }
                        return false;
                    });
                
                for (int64_t i = 0; i < last_dim; ++i) {
                    dst[batch * last_dim + i] = indices[i];
                }
            }
            break;
        }
        case INSIGHT_DTYPE_F64: {
            const double* src = (const double*)transposed->data;
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                std::vector<std::vector<double>> batch_keys(nkeys);
                for (int64_t k = 0; k < nkeys; ++k) {
                    batch_keys[k].resize(last_dim);
                    const double* key_start = src + (batch * nkeys + k) * last_dim;
                    for (int64_t i = 0; i < last_dim; ++i) {
                        batch_keys[k][i] = key_start[i];
                    }
                }
                
                std::vector<int64_t> indices(last_dim);
                for (int64_t i = 0; i < last_dim; ++i) indices[i] = i;
                
                std::sort(indices.begin(), indices.end(),
                    [&](int64_t a, int64_t b) {
                        for (int64_t k = 0; k < nkeys; ++k) {
                            if (batch_keys[k][a] < batch_keys[k][b]) return true;
                            if (batch_keys[k][a] > batch_keys[k][b]) return false;
                        }
                        return false;
                    });
                
                for (int64_t i = 0; i < last_dim; ++i) {
                    dst[batch * last_dim + i] = indices[i];
                }
            }
            break;
        }
        case INSIGHT_DTYPE_I32: {
            const int32_t* src = (const int32_t*)transposed->data;
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                std::vector<std::vector<int32_t>> batch_keys(nkeys);
                for (int64_t k = 0; k < nkeys; ++k) {
                    batch_keys[k].resize(last_dim);
                    const int32_t* key_start = src + (batch * nkeys + k) * last_dim;
                    for (int64_t i = 0; i < last_dim; ++i) {
                        batch_keys[k][i] = key_start[i];
                    }
                }
                
                std::vector<int64_t> indices(last_dim);
                for (int64_t i = 0; i < last_dim; ++i) indices[i] = i;
                
                std::sort(indices.begin(), indices.end(),
                    [&](int64_t a, int64_t b) {
                        for (int64_t k = 0; k < nkeys; ++k) {
                            if (batch_keys[k][a] < batch_keys[k][b]) return true;
                            if (batch_keys[k][a] > batch_keys[k][b]) return false;
                        }
                        return false;
                    });
                
                for (int64_t i = 0; i < last_dim; ++i) {
                    dst[batch * last_dim + i] = indices[i];
                }
            }
            break;
        }
        case INSIGHT_DTYPE_I64: {
            const int64_t* src = (const int64_t*)transposed->data;
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                std::vector<std::vector<int64_t>> batch_keys(nkeys);
                for (int64_t k = 0; k < nkeys; ++k) {
                    batch_keys[k].resize(last_dim);
                    const int64_t* key_start = src + (batch * nkeys + k) * last_dim;
                    for (int64_t i = 0; i < last_dim; ++i) {
                        batch_keys[k][i] = key_start[i];
                    }
                }
                
                std::vector<int64_t> indices(last_dim);
                for (int64_t i = 0; i < last_dim; ++i) indices[i] = i;
                
                std::sort(indices.begin(), indices.end(),
                    [&](int64_t a, int64_t b) {
                        for (int64_t k = 0; k < nkeys; ++k) {
                            if (batch_keys[k][a] < batch_keys[k][b]) return true;
                            if (batch_keys[k][a] > batch_keys[k][b]) return false;
                        }
                        return false;
                    });
                
                for (int64_t i = 0; i < last_dim; ++i) {
                    dst[batch * last_dim + i] = indices[i];
                }
            }
            break;
        }
        default:
            cpu_set_last_error("lexsort: unsupported dtype");
            return C_FAILED;
    }
    
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(lexsort, INSIGHT_DTYPE_F32, lexsort_kernel_cpu);
REGISTER_CPU_KERNEL(lexsort, INSIGHT_DTYPE_F64, lexsort_kernel_cpu);
REGISTER_CPU_KERNEL(lexsort, INSIGHT_DTYPE_I32, lexsort_kernel_cpu);
REGISTER_CPU_KERNEL(lexsort, INSIGHT_DTYPE_I64, lexsort_kernel_cpu);
