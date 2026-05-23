// backends/cpu/kernels/cast/cast.cpp
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include "insight/c_api/dtype.h"
#include <cstring>
#include <complex>
#include <cmath>

/**
 * @brief CPU kernel for cast operation.
 *
 * Converts elements from source dtype to target dtype.
 * The kernel is dispatched by the source dtype (inputs[0]->dtype).
 * The target dtype is passed as int32_t* in inputs[1].
 * Output array is pre-allocated by the frontend.
 *
 * @param inputs  [0] = InsightArray* source, [1] = int32_t* target_dtype
 * @param outputs [0] = InsightArray* destination (pre-allocated)
 * @return C_SUCCESS on success, C_FAILED on error
 */
extern "C" {

    C_Status cast_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* src = static_cast<InsightArray*>(inputs[0]);
        InsightArray* dst = static_cast<InsightArray*>(outputs[0]);

        if (!src || !dst) {
            cpu_set_last_error("cast: source or destination array is null");
            return C_FAILED;
        }

        if (!inputs[1]) {
            cpu_set_last_error("cast: target_dtype is null");
            return C_FAILED;
        }

        int32_t target_dtype = *static_cast<int32_t*>(inputs[1]);
        int64_t n = src->numel;

        // Sanity check: shapes must match
        if (src->numel != dst->numel) {
            cpu_set_last_error("cast: source and destination numel mismatch");
            return C_FAILED;
        }

        // Fast path: same dtype (should be handled by frontend, but just in case)
        if (src->dtype == target_dtype) {
            std::memcpy(dst->data, src->data, n * insight_dtype_size(src->dtype));
            return C_SUCCESS;
        }

        // Dispatch based on source dtype and target dtype
        switch (src->dtype) {
            // ========== From BOOL ==========
        case INSIGHT_DTYPE_BOOL: {
            const bool* s = static_cast<const bool*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_U8: {
                uint8_t* d = static_cast<uint8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] ? 1 : 0;
                break;
            }
            case INSIGHT_DTYPE_I8: {
                int8_t* d = static_cast<int8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] ? 1 : 0;
                break;
            }
            case INSIGHT_DTYPE_I16: {
                int16_t* d = static_cast<int16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] ? 1 : 0;
                break;
            }
            case INSIGHT_DTYPE_I32: {
                int32_t* d = static_cast<int32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] ? 1 : 0;
                break;
            }
            case INSIGHT_DTYPE_I64: {
                int64_t* d = static_cast<int64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] ? 1 : 0;
                break;
            }
            case INSIGHT_DTYPE_U16: {
                uint16_t* d = static_cast<uint16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] ? 1 : 0;
                break;
            }
            case INSIGHT_DTYPE_U32: {
                uint32_t* d = static_cast<uint32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] ? 1 : 0;
                break;
            }
            case INSIGHT_DTYPE_U64: {
                uint64_t* d = static_cast<uint64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] ? 1 : 0;
                break;
            }
            case INSIGHT_DTYPE_F32: {
                float* d = static_cast<float*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] ? 1.0f : 0.0f;
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* d = static_cast<double*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] ? 1.0 : 0.0;
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = s[i] ? std::complex<float>(1.0f, 0.0f) : std::complex<float>(0.0f, 0.0f);
                }
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = s[i] ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from bool");
                return C_FAILED;
            }
            break;
        }

                               // ========== From U8 ==========
        case INSIGHT_DTYPE_U8: {
            const uint8_t* s = static_cast<const uint8_t*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] != 0;
                break;
            }
            case INSIGHT_DTYPE_I8: {
                int8_t* d = static_cast<int8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I16: {
                int16_t* d = static_cast<int16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I32: {
                int32_t* d = static_cast<int32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I64: {
                int64_t* d = static_cast<int64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U16: {
                uint16_t* d = static_cast<uint16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U32: {
                uint32_t* d = static_cast<uint32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U64: {
                uint64_t* d = static_cast<uint64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F32: {
                float* d = static_cast<float*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* d = static_cast<double*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<double>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<float>(static_cast<float>(s[i]), 0.0f);
                }
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<double>(static_cast<double>(s[i]), 0.0);
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from uint8");
                return C_FAILED;
            }
            break;
        }

                             // ========== From I8 ==========
        case INSIGHT_DTYPE_I8: {
            const int8_t* s = static_cast<const int8_t*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] != 0;
                break;
            }
            case INSIGHT_DTYPE_U8: {
                uint8_t* d = static_cast<uint8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I16: {
                int16_t* d = static_cast<int16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I32: {
                int32_t* d = static_cast<int32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I64: {
                int64_t* d = static_cast<int64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U16: {
                uint16_t* d = static_cast<uint16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U32: {
                uint32_t* d = static_cast<uint32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U64: {
                uint64_t* d = static_cast<uint64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F32: {
                float* d = static_cast<float*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* d = static_cast<double*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<double>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<float>(static_cast<float>(s[i]), 0.0f);
                }
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<double>(static_cast<double>(s[i]), 0.0);
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from int8");
                return C_FAILED;
            }
            break;
        }

                             // ========== From I16 ==========
        case INSIGHT_DTYPE_I16: {
            const int16_t* s = static_cast<const int16_t*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] != 0;
                break;
            }
            case INSIGHT_DTYPE_U8: {
                uint8_t* d = static_cast<uint8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I8: {
                int8_t* d = static_cast<int8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I32: {
                int32_t* d = static_cast<int32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I64: {
                int64_t* d = static_cast<int64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U16: {
                uint16_t* d = static_cast<uint16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U32: {
                uint32_t* d = static_cast<uint32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U64: {
                uint64_t* d = static_cast<uint64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F32: {
                float* d = static_cast<float*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* d = static_cast<double*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<double>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<float>(static_cast<float>(s[i]), 0.0f);
                }
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<double>(static_cast<double>(s[i]), 0.0);
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from int16");
                return C_FAILED;
            }
            break;
        }

                              // ========== From I32 ==========
        case INSIGHT_DTYPE_I32: {
            const int32_t* s = static_cast<const int32_t*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] != 0;
                break;
            }
            case INSIGHT_DTYPE_U8: {
                uint8_t* d = static_cast<uint8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I8: {
                int8_t* d = static_cast<int8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I16: {
                int16_t* d = static_cast<int16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I64: {
                int64_t* d = static_cast<int64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U16: {
                uint16_t* d = static_cast<uint16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U32: {
                uint32_t* d = static_cast<uint32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U64: {
                uint64_t* d = static_cast<uint64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F32: {
                float* d = static_cast<float*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* d = static_cast<double*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<double>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<float>(static_cast<float>(s[i]), 0.0f);
                }
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<double>(static_cast<double>(s[i]), 0.0);
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from int32");
                return C_FAILED;
            }
            break;
        }

                              // ========== From I64 ==========
        case INSIGHT_DTYPE_I64: {
            const int64_t* s = static_cast<const int64_t*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] != 0;
                break;
            }
            case INSIGHT_DTYPE_U8: {
                uint8_t* d = static_cast<uint8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I8: {
                int8_t* d = static_cast<int8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I16: {
                int16_t* d = static_cast<int16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I32: {
                int32_t* d = static_cast<int32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U16: {
                uint16_t* d = static_cast<uint16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U32: {
                uint32_t* d = static_cast<uint32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U64: {
                uint64_t* d = static_cast<uint64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F32: {
                float* d = static_cast<float*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* d = static_cast<double*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<double>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<float>(static_cast<float>(s[i]), 0.0f);
                }
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<double>(static_cast<double>(s[i]), 0.0);
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from int64");
                return C_FAILED;
            }
            break;
        }

                              // ========== From U16 ==========
        case INSIGHT_DTYPE_U16: {
            const uint16_t* s = static_cast<const uint16_t*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] != 0;
                break;
            }
            case INSIGHT_DTYPE_U8: {
                uint8_t* d = static_cast<uint8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I8: {
                int8_t* d = static_cast<int8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I16: {
                int16_t* d = static_cast<int16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I32: {
                int32_t* d = static_cast<int32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I64: {
                int64_t* d = static_cast<int64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U32: {
                uint32_t* d = static_cast<uint32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U64: {
                uint64_t* d = static_cast<uint64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F32: {
                float* d = static_cast<float*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* d = static_cast<double*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<double>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<float>(static_cast<float>(s[i]), 0.0f);
                }
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<double>(static_cast<double>(s[i]), 0.0);
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from uint16");
                return C_FAILED;
            }
            break;
        }

                              // ========== From U32 ==========
        case INSIGHT_DTYPE_U32: {
            const uint32_t* s = static_cast<const uint32_t*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] != 0;
                break;
            }
            case INSIGHT_DTYPE_U8: {
                uint8_t* d = static_cast<uint8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I8: {
                int8_t* d = static_cast<int8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I16: {
                int16_t* d = static_cast<int16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I32: {
                int32_t* d = static_cast<int32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I64: {
                int64_t* d = static_cast<int64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U16: {
                uint16_t* d = static_cast<uint16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U64: {
                uint64_t* d = static_cast<uint64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F32: {
                float* d = static_cast<float*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* d = static_cast<double*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<double>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<float>(static_cast<float>(s[i]), 0.0f);
                }
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<double>(static_cast<double>(s[i]), 0.0);
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from uint32");
                return C_FAILED;
            }
            break;
        }

                              // ========== From U64 ==========
        case INSIGHT_DTYPE_U64: {
            const uint64_t* s = static_cast<const uint64_t*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] != 0;
                break;
            }
            case INSIGHT_DTYPE_U8: {
                uint8_t* d = static_cast<uint8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I8: {
                int8_t* d = static_cast<int8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I16: {
                int16_t* d = static_cast<int16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I32: {
                int32_t* d = static_cast<int32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I64: {
                int64_t* d = static_cast<int64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U16: {
                uint16_t* d = static_cast<uint16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U32: {
                uint32_t* d = static_cast<uint32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F32: {
                float* d = static_cast<float*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* d = static_cast<double*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<double>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<float>(static_cast<float>(s[i]), 0.0f);
                }
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<double>(static_cast<double>(s[i]), 0.0);
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from uint64");
                return C_FAILED;
            }
            break;
        }

                              // ========== From F32 ==========
        case INSIGHT_DTYPE_F32: {
            const float* s = static_cast<const float*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] != 0.0f;
                break;
            }
            case INSIGHT_DTYPE_U8: {
                uint8_t* d = static_cast<uint8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I8: {
                int8_t* d = static_cast<int8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I16: {
                int16_t* d = static_cast<int16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I32: {
                int32_t* d = static_cast<int32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I64: {
                int64_t* d = static_cast<int64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U16: {
                uint16_t* d = static_cast<uint16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U32: {
                uint32_t* d = static_cast<uint32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U64: {
                uint64_t* d = static_cast<uint64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* d = static_cast<double*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<double>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<float>(s[i], 0.0f);
                }
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<double>(static_cast<double>(s[i]), 0.0);
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from float32");
                return C_FAILED;
            }
            break;
        }

                              // ========== From F64 ==========
        case INSIGHT_DTYPE_F64: {
            const double* s = static_cast<const double*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i] != 0.0;
                break;
            }
            case INSIGHT_DTYPE_U8: {
                uint8_t* d = static_cast<uint8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I8: {
                int8_t* d = static_cast<int8_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I16: {
                int16_t* d = static_cast<int16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I32: {
                int32_t* d = static_cast<int32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_I64: {
                int64_t* d = static_cast<int64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U16: {
                uint16_t* d = static_cast<uint16_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint16_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U32: {
                uint32_t* d = static_cast<uint32_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint32_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_U64: {
                uint64_t* d = static_cast<uint64_t*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<uint64_t>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_F32: {
                float* d = static_cast<float*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<float>(static_cast<float>(s[i]), 0.0f);
                }
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<double>(s[i], 0.0);
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from float64");
                return C_FAILED;
            }
            break;
        }

                              // ========== From C32 ==========
        case INSIGHT_DTYPE_C32: {
            const std::complex<float>* s = static_cast<const std::complex<float>*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i].real() != 0.0f || s[i].imag() != 0.0f;
                break;
            }
            case INSIGHT_DTYPE_F32: {
                float* d = static_cast<float*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i].real();
                break;
            }
            case INSIGHT_DTYPE_C64: {
                std::complex<double>* d = static_cast<std::complex<double>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<double>(static_cast<double>(s[i].real()),
                        static_cast<double>(s[i].imag()));
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from complex64");
                return C_FAILED;
            }
            break;
        }

                              // ========== From C64 ==========
        case INSIGHT_DTYPE_C64: {
            const std::complex<double>* s = static_cast<const std::complex<double>*>(src->data);
            switch (target_dtype) {
            case INSIGHT_DTYPE_BOOL: {
                bool* d = static_cast<bool*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i].real() != 0.0 || s[i].imag() != 0.0;
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* d = static_cast<double*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) d[i] = s[i].real();
                break;
            }
            case INSIGHT_DTYPE_C32: {
                std::complex<float>* d = static_cast<std::complex<float>*>(dst->data);
                #pragma omp parallel for
                for (int64_t i = 0; i < n; ++i) {
                    d[i] = std::complex<float>(static_cast<float>(s[i].real()),
                        static_cast<float>(s[i].imag()));
                }
                break;
            }
            default:
                cpu_set_last_error("cast: unsupported target dtype from complex128");
                return C_FAILED;
            }
            break;
        }

        default:
            cpu_set_last_error(("cast: unsupported source dtype " + std::to_string(src->dtype)).c_str());
            return C_FAILED;
        }

        return C_SUCCESS;
    }

} // extern "C"

// Register cast kernel for all source dtypes
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_BOOL, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_U8, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_I8, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_I16, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_I32, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_I64, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_U16, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_U32, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_U64, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_F32, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_F64, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_C32, cast_kernel_cpu);
REGISTER_CPU_KERNEL(cast, INSIGHT_DTYPE_C64, cast_kernel_cpu);