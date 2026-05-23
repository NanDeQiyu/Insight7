// backends/cpu/kernels/indexing/compress.cpp
/**
 * @file compress.cpp
 * @brief CPU kernel for compress operation.
 *
 * Selects slices along an axis where condition is true.
 */

#include "common.h"
#include <complex>



 // 通用模板实现
template<typename T>
static C_Status compress_impl(
    InsightArray* out,
    InsightArray* x,
    InsightArray* cond,
    int axis) {

    int64_t ndim = out->ndim;
    int64_t dims[INSIGHT_MAX_NDIM];
    int64_t x_strides[INSIGHT_MAX_NDIM];
    int64_t out_strides[INSIGHT_MAX_NDIM];

    for (int i = 0; i < ndim; ++i) {
        dims[i] = out->dims[i];
        x_strides[i] = x->strides[i];
        out_strides[i] = out->strides[i];
    }

    int64_t axis_dim = x->dims[axis];
    int64_t keep_count = out->dims[axis];
    const bool* cond_data = (const bool*)cond->data;

    // 1. 构建正向映射：condition_index -> output_index
    int64_t* fwd_map = (int64_t*)malloc(axis_dim * sizeof(int64_t));
    if (!fwd_map) {
        cpu_set_last_error("compress: memory allocation failed");
        return C_FAILED;
    }

    // 2. 构建反向映射：output_index -> condition_index (O(1) 查找)
    int64_t* rev_map = (int64_t*)malloc(keep_count * sizeof(int64_t));
    if (!rev_map) {
        free(fwd_map);
        cpu_set_last_error("compress: memory allocation failed");
        return C_FAILED;
    }

    int64_t out_axis_idx = 0;
    for (int64_t i = 0; i < axis_dim; ++i) {
        if (cond_data[i]) {
            fwd_map[i] = out_axis_idx;
            rev_map[out_axis_idx] = i;
            out_axis_idx++;
        }
        else {
            fwd_map[i] = -1;
        }
    }

    int64_t total_out = out->numel;
    T* dst = (T*)out->data;
    const T* src = (const T*)x->data;

#pragma omp parallel for
    for (int64_t linear = 0; linear < total_out; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        cpu_linear_to_indices(linear, ndim, dims, indices);

        // 计算输出 offset
        int64_t dst_offset = out->offset;
        for (int d = 0; d < ndim; ++d) {
            dst_offset += indices[d] * out_strides[d];
        }

        // 计算输入 offset
        int64_t src_offset = x->offset;
        for (int d = 0; d < ndim; ++d) {
            if (d == axis) {
                // O(1) 查找：从 output index 得到 condition index
                int64_t orig_idx = rev_map[indices[d]];
                src_offset += orig_idx * x_strides[d];
            }
            else {
                src_offset += indices[d] * x_strides[d];
            }
        }

        dst[dst_offset] = src[src_offset];
    }

    free(fwd_map);
    free(rev_map);
    return C_SUCCESS;
}

// 类型分发宏
#define COMPRESS_CASE(DTYPE, CTYPE) \
    case DTYPE: \
        return compress_impl<CTYPE>(out, x, cond, axis);

#ifdef __cplusplus
extern "C" {
#endif

    C_Status compress_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* out = (InsightArray*)outputs[0];
        InsightArray* x = (InsightArray*)inputs[1];
        InsightArray* cond = (InsightArray*)inputs[2];
        int axis = *(int*)inputs[3];

        if (!out || !x || !cond) {
            cpu_set_last_error("compress: null array pointer");
            return C_FAILED;
        }

        // 确保 condition 是 bool 类型
        if (cond->dtype != INSIGHT_DTYPE_BOOL) {
            cpu_set_last_error("compress: condition must be bool");
            return C_FAILED;
        }

        // 确保 condition 是 1D
        if (cond->ndim != 1) {
            cpu_set_last_error("compress: condition must be 1D");
            return C_FAILED;
        }

        // 确保 condition 长度等于 axis 维度
        if (cond->dims[0] != x->dims[axis]) {
            cpu_set_last_error("compress: condition length must match axis dimension");
            return C_FAILED;
        }

        switch (x->dtype) {
            COMPRESS_CASE(INSIGHT_DTYPE_BOOL, bool)
                COMPRESS_CASE(INSIGHT_DTYPE_U8, uint8_t)
                COMPRESS_CASE(INSIGHT_DTYPE_I8, int8_t)
                COMPRESS_CASE(INSIGHT_DTYPE_I16, int16_t)
                COMPRESS_CASE(INSIGHT_DTYPE_I32, int32_t)
                COMPRESS_CASE(INSIGHT_DTYPE_I64, int64_t)
                COMPRESS_CASE(INSIGHT_DTYPE_U16, uint16_t)
                COMPRESS_CASE(INSIGHT_DTYPE_U32, uint32_t)
                COMPRESS_CASE(INSIGHT_DTYPE_U64, uint64_t)
                COMPRESS_CASE(INSIGHT_DTYPE_F16, uint16_t)
                COMPRESS_CASE(INSIGHT_DTYPE_BF16, uint16_t)
                COMPRESS_CASE(INSIGHT_DTYPE_F32, float)
                COMPRESS_CASE(INSIGHT_DTYPE_F64, double)
                COMPRESS_CASE(INSIGHT_DTYPE_C32, std::complex<float>)
                COMPRESS_CASE(INSIGHT_DTYPE_C64, std::complex<double>)
        default:
            cpu_set_last_error("compress: unsupported dtype");
            return C_FAILED;
        }
    }

#ifdef __cplusplus
}
#endif

// 注册所有类型
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_BOOL, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_U8, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_I8, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_I16, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_I32, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_I64, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_U16, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_U32, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_U64, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_F16, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_BF16, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_F32, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_F64, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_C32, compress_kernel_cpu);
REGISTER_CPU_KERNEL(compress, INSIGHT_DTYPE_C64, compress_kernel_cpu);