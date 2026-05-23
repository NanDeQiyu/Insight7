// backends/cpu/kernels/indexing/searchsorted.cpp
/**
 * @file searchsorted.cpp
 * @brief CPU kernel for searchsorted operation.
 *
 * Finds indices where elements should be inserted to maintain order.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (indices)
 *   inputs[1] = InsightArray* x (sorted 1D)
 *   inputs[2] = InsightArray* v (values)
 *   inputs[3] = int* side_code (0 = left, 1 = right)
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

template<typename T>
static int64_t binary_search(const T* x, int64_t n, T val, bool is_left) {
    int64_t lo = 0;
    int64_t hi = n;

    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (is_left) {
            if (x[mid] < val) {
                lo = mid + 1;
            }
            else {
                hi = mid;
            }
        }
        else {
            if (x[mid] <= val) {
                lo = mid + 1;
            }
            else {
                hi = mid;
            }
        }
    }

    return lo;
}

#define SEARCHSORTED_DISPATCH(CTYPE) \
    do { \
        const CTYPE* x_data = (const CTYPE*)x->data; \
        const CTYPE* v_data = (const CTYPE*)v->data; \
        for (int64_t i = 0; i < n; ++i) { \
            dst[i] = binary_search(x_data, m, v_data[i], is_left); \
        } \
    } while(0)

#ifdef __cplusplus
extern "C" {
#endif

    C_Status searchsorted_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* out = (InsightArray*)outputs[0];
        InsightArray* x = (InsightArray*)inputs[1];
        InsightArray* v = (InsightArray*)inputs[2];
        int side_code = *(int*)inputs[3];

        if (!out || !x || !v) {
            cpu_set_last_error("searchsorted: null array pointer");
            return C_FAILED;
        }

        int64_t n = v->numel;
        int64_t m = x->numel;
        int64_t* dst = (int64_t*)out->data;
        bool is_left = (side_code == 0);

        switch (x->dtype) {
        case INSIGHT_DTYPE_BOOL:   SEARCHSORTED_DISPATCH(bool); break;
        case INSIGHT_DTYPE_U8:     SEARCHSORTED_DISPATCH(uint8_t); break;
        case INSIGHT_DTYPE_U16:    SEARCHSORTED_DISPATCH(uint16_t); break;
        case INSIGHT_DTYPE_U32:    SEARCHSORTED_DISPATCH(uint32_t); break;
        case INSIGHT_DTYPE_U64:    SEARCHSORTED_DISPATCH(uint64_t); break;
        case INSIGHT_DTYPE_I8:     SEARCHSORTED_DISPATCH(int8_t); break;
        case INSIGHT_DTYPE_I16:    SEARCHSORTED_DISPATCH(int16_t); break;
        case INSIGHT_DTYPE_I32:    SEARCHSORTED_DISPATCH(int32_t); break;
        case INSIGHT_DTYPE_I64:    SEARCHSORTED_DISPATCH(int64_t); break;
        case INSIGHT_DTYPE_F16:    SEARCHSORTED_DISPATCH(uint16_t); break;
        case INSIGHT_DTYPE_BF16:   SEARCHSORTED_DISPATCH(uint16_t); break;
        case INSIGHT_DTYPE_F32:    SEARCHSORTED_DISPATCH(float); break;
        case INSIGHT_DTYPE_F64:    SEARCHSORTED_DISPATCH(double); break;
        case INSIGHT_DTYPE_F8_E4M3: SEARCHSORTED_DISPATCH(uint8_t); break;
        case INSIGHT_DTYPE_F8_E5M2: SEARCHSORTED_DISPATCH(uint8_t); break;
        default:
            cpu_set_last_error("searchsorted: unsupported dtype");
            return C_FAILED;
        }

        return C_SUCCESS;
    }

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_BOOL, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_U8, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_U16, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_U32, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_U64, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_I8, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_I16, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_I32, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_I64, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_F16, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_BF16, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_F32, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_F64, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_F8_E4M3, searchsorted_kernel_cpu);
REGISTER_CPU_KERNEL(searchsorted, INSIGHT_DTYPE_F8_E5M2, searchsorted_kernel_cpu);