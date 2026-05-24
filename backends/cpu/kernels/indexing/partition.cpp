// backends/cpu/kernels/indexing/partition.cpp
/**
 * @file partition.cpp
 * @brief CPU kernel for partition operation.
 *
 * Partially sorts array so that the k-th element is in its sorted position.
 * Supports both 1D and multi-dimensional arrays.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output
 *   inputs[1] = InsightArray* input
 *   inputs[2] = int64_t* kth
 *   inputs[3] = int* axis
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <algorithm>
#include <vector>
#include <cstring>


template<typename T>
static void partition_1d_impl(const T* src, T* dst, int64_t n, int64_t kth) {
    std::vector<T> data(src, src + n);
    std::nth_element(data.begin(), data.begin() + kth, data.end());
    std::memcpy(dst, data.data(), n * sizeof(T));
}

template<>
void partition_1d_impl<bool>(const bool* src, bool* dst, int64_t n, int64_t kth) {
    std::vector<bool> data(src, src + n);
    std::nth_element(data.begin(), data.begin() + kth, data.end());
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = data[i];
    }
}

template<typename T>
static void partition_nd_impl(
    const T* src,
    T* dst,
    int64_t ndim,
    const int64_t* dims,
    const int64_t* src_strides,
    const int64_t* dst_strides,
    int axis,
    int64_t kth) {

    // 计算 axis 维度的步长
    int64_t axis_stride = 1;
    for (int d = axis + 1; d < ndim; ++d) {
        axis_stride *= dims[d];
    }

    int64_t axis_size = dims[axis];
    int64_t batch_size = 1;
    for (int i = 0; i < ndim; ++i) {
        if (i != axis) batch_size *= dims[i];
    }

    // 预计算每个批次的基地址
    std::vector<int64_t> batch_offsets(batch_size);
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        int64_t base_offset = 0;
        int64_t tmp = batch;
        for (int d = ndim - 1; d >= 0; --d) {
            if (d == axis) continue;
            int64_t coord = tmp % dims[d];
            tmp /= dims[d];
            base_offset += coord * dst_strides[d];
        }
        batch_offsets[batch] = base_offset;
    }

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        // 收集这个 slice 的数据
        std::vector<T> slice_data(axis_size);
        int64_t base_offset = 0;
        int64_t tmp = batch;
        for (int d = ndim - 1; d >= 0; --d) {
            if (d == axis) continue;
            int64_t coord = tmp % dims[d];
            tmp /= dims[d];
            base_offset += coord * src_strides[d];
        }

        for (int64_t i = 0; i < axis_size; ++i) {
            int64_t src_offset = base_offset + i * src_strides[axis];
            slice_data[i] = src[src_offset];
        }

        // 分区
        std::nth_element(slice_data.begin(), slice_data.begin() + kth, slice_data.end());

        // 写回
        for (int64_t i = 0; i < axis_size; ++i) {
            int64_t dst_offset = batch_offsets[batch] + i * dst_strides[axis];
            dst[dst_offset] = slice_data[i];
        }
    }
}

#define PARTITION_DISPATCH(CTYPE) \
    do { \
        if (ndim == 1) { \
            partition_1d_impl<CTYPE>((const CTYPE*)x->data, (CTYPE*)out->data, n, kth); \
        } else { \
            partition_nd_impl<CTYPE>( \
                (const CTYPE*)x->data, (CTYPE*)out->data, \
                ndim, dims, src_strides, dst_strides, axis, kth); \
        } \
    } while(0)

#ifdef __cplusplus
extern "C" {
#endif

    C_Status partition_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* out = (InsightArray*)outputs[0];
        InsightArray* x = (InsightArray*)inputs[1];
        int64_t kth = *(int64_t*)inputs[2];
        int axis = *(int*)inputs[3];

        if (!out || !x) {
            cpu_set_last_error("partition: null array pointer");
            return C_FAILED;
        }

        int64_t ndim = x->ndim;
        if (axis < 0) axis += ndim;
        if (axis < 0 || axis >= ndim) {
            cpu_set_last_error("partition: axis out of range");
            return C_FAILED;
        }

        int64_t n = x->numel;
        int64_t dims[INSIGHT_MAX_NDIM];
        int64_t src_strides[INSIGHT_MAX_NDIM];
        int64_t dst_strides[INSIGHT_MAX_NDIM];

        for (int i = 0; i < ndim; ++i) {
            dims[i] = out->dims[i];
            src_strides[i] = x->strides[i];
            dst_strides[i] = out->strides[i];
        }

        switch (x->dtype) {
        case INSIGHT_DTYPE_BOOL:   PARTITION_DISPATCH(bool); break;
        case INSIGHT_DTYPE_U8:     PARTITION_DISPATCH(uint8_t); break;
        case INSIGHT_DTYPE_U16:    PARTITION_DISPATCH(uint16_t); break;
        case INSIGHT_DTYPE_U32:    PARTITION_DISPATCH(uint32_t); break;
        case INSIGHT_DTYPE_U64:    PARTITION_DISPATCH(uint64_t); break;
        case INSIGHT_DTYPE_I8:     PARTITION_DISPATCH(int8_t); break;
        case INSIGHT_DTYPE_I16:    PARTITION_DISPATCH(int16_t); break;
        case INSIGHT_DTYPE_I32:    PARTITION_DISPATCH(int32_t); break;
        case INSIGHT_DTYPE_I64:    PARTITION_DISPATCH(int64_t); break;
        case INSIGHT_DTYPE_F16:    PARTITION_DISPATCH(uint16_t); break;
        case INSIGHT_DTYPE_BF16:   PARTITION_DISPATCH(uint16_t); break;
        case INSIGHT_DTYPE_F32:    PARTITION_DISPATCH(float); break;
        case INSIGHT_DTYPE_F64:    PARTITION_DISPATCH(double); break;
        case INSIGHT_DTYPE_F8_E4M3: PARTITION_DISPATCH(uint8_t); break;
        case INSIGHT_DTYPE_F8_E5M2: PARTITION_DISPATCH(uint8_t); break;
        default:
            cpu_set_last_error("partition: unsupported dtype");
            return C_FAILED;
        }

        return C_SUCCESS;
    }

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_BOOL, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_U8, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_U16, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_U32, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_U64, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_I8, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_I16, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_I32, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_I64, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_F16, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_BF16, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_F32, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_F64, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_F8_E4M3, partition_kernel_cpu);
REGISTER_CPU_KERNEL(partition, INSIGHT_DTYPE_F8_E5M2, partition_kernel_cpu);