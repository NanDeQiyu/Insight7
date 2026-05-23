// backends/cpu/kernels/manipulation/contiguous_copy.cpp
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include "insight/core/dtype.h"
#include <cstring>

extern "C" {

    C_Status contiguous_copy_cpu(void** inputs, void** outputs) {
        InsightArray* in = static_cast<InsightArray*>(inputs[0]);
        InsightArray* out = static_cast<InsightArray*>(outputs[0]);

        if (!in) {
            cpu_set_last_error("contiguous_copy: input array is null");
            return C_FAILED;
        }
        if (!out) {
            cpu_set_last_error("contiguous_copy: output array is null");
            return C_FAILED;
        }
        if (!in->data) {
            cpu_set_last_error("contiguous_copy: input array has no data");
            return C_FAILED;
        }
        if (!out->data) {
            cpu_set_last_error("contiguous_copy: output array has no data");
            return C_FAILED;
        }

        size_t elem_size = insight_dtype_size(in->dtype);
        if (elem_size == 0) {
            cpu_set_last_error("contiguous_copy: unknown data type");
            return C_FAILED;
        }

        int64_t total = in->numel;
        int ndim = in->ndim;

        // 一维数组和标量直接 memcpy
        if (ndim <= 1 || in->offset == 0) {
            bool is_contiguous_in = true;
            int64_t expected_stride = 1;
            for (int d = ndim - 1; d >= 0; --d) {
                if (in->strides[d] != expected_stride) {
                    is_contiguous_in = false;
                    break;
                }
                expected_stride *= in->dims[d];
            }
            if (is_contiguous_in) {
                std::memcpy(out->data, in->data, total * elem_size);
                return C_SUCCESS;
            }
        }

        // 非连续数组逐元素复制
        for (int64_t linear = 0; linear < total; ++linear) {
            int64_t in_offset = in->offset;
            int64_t tmp = linear;
            for (int d = ndim - 1; d >= 0; --d) {
                int64_t idx = tmp % in->dims[d];
                in_offset += idx * in->strides[d];
                tmp /= in->dims[d];
            }

            char* in_ptr = static_cast<char*>(in->data) + in_offset * elem_size;
            char* out_ptr = static_cast<char*>(out->data) + linear * elem_size;
            std::memcpy(out_ptr, in_ptr, elem_size);
        }

        return C_SUCCESS;
    }

} // extern "C"

// Register for ALL supported types
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_BOOL, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_U8, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_I8, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_I16, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_I32, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_I64, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_U16, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_U32, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_U64, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_F16, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_BF16, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_F32, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_F64, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_C32, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_C64, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_F8_E4M3, contiguous_copy_cpu);
REGISTER_CPU_KERNEL(contiguous_copy, INSIGHT_DTYPE_F8_E5M2, contiguous_copy_cpu);