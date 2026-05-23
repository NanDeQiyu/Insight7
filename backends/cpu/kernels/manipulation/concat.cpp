// backends/cpu/kernels/manipulation/concat.cpp
/**
 * @file concat.cpp
 * @brief CPU kernel for concatenation operation.
 *
 * Concatenates multiple arrays along an axis.
 *
 * Input layout:
 *   inputs[0] = output array (unused, outputs[0] is used instead)
 *   inputs[1] = int* num_inputs (number of input arrays)
 *   inputs[2] = InsightArray* input0
 *   inputs[3] = InsightArray* input1
 *   ...
 *   inputs[2 + num_inputs] = int* axis
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <vector>
#include <complex>
#include <cstring>


template<typename T>
static void concat_impl(
    const std::vector<const InsightArray*>& inputs,
    T* dst,
    int64_t ndim,
    const int64_t* out_dims,
    const int64_t* out_strides,
    int axis) {

    // Compute cumulative sizes along the concatenation axis
    std::vector<int64_t> cum_sizes;
    cum_sizes.push_back(0);
    for (const auto* arr : inputs) {
        cum_sizes.push_back(cum_sizes.back() + arr->dims[axis]);
    }

    int64_t total_out = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total_out *= out_dims[d];
    }

#pragma omp parallel for
    for (int64_t linear = 0; linear < total_out; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        cpu_linear_to_indices(linear, ndim, out_dims, indices);

        // Find which input this index belongs to
        int64_t axis_idx = indices[axis];
        int tensor_idx = 0;
        for (size_t i = 1; i < cum_sizes.size(); ++i) {
            if (axis_idx < cum_sizes[i]) {
                tensor_idx = static_cast<int>(i - 1);
                break;
            }
        }

        const InsightArray* src_arr = inputs[tensor_idx];
        const T* src = (const T*)src_arr->data;
        int64_t src_offset = src_arr->offset;
        for (int64_t d = 0; d < ndim; ++d) {
            int64_t idx;
            if (d == axis) {
                idx = axis_idx - cum_sizes[tensor_idx];
            }
            else {
                idx = indices[d];
            }
            src_offset += idx * src_arr->strides[d];
        }

        dst[linear] = src[src_offset];
    }
}

#ifdef __cplusplus
extern "C" {
#endif

    C_Status concat_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* out = (InsightArray*)outputs[0];

        if (!out) {
            cpu_set_last_error("concat: output is null");
            return C_FAILED;
        }

        if (!inputs[1]) {
            cpu_set_last_error("concat: num_inputs is null");
            return C_FAILED;
        }

        int num_inputs = *(int*)inputs[1];

        if (num_inputs < 1) {
            cpu_set_last_error("concat: no input arrays");
            return C_FAILED;
        }

        // Extract input arrays
        std::vector<const InsightArray*> in_arrays;
        in_arrays.reserve(num_inputs);
        for (int i = 0; i < num_inputs; ++i) {
            if (!inputs[2 + i]) {
                cpu_set_last_error("concat: null input array");
                return C_FAILED;
            }
            in_arrays.push_back((InsightArray*)inputs[2 + i]);
        }

        // Extract axis
        if (!inputs[2 + num_inputs]) {
            cpu_set_last_error("concat: axis is null");
            return C_FAILED;
        }
        int axis = *(int*)inputs[2 + num_inputs];

        int64_t ndim = out->ndim;
        int64_t out_dims[INSIGHT_MAX_NDIM];
        int64_t out_strides[INSIGHT_MAX_NDIM];
        for (int i = 0; i < ndim; ++i) {
            out_dims[i] = out->dims[i];
        }
        cpu_compute_strides(ndim, out_dims, out_strides);

        // Ensure output is contiguous (should be by frontend)
        if (!insight_array_is_contiguous(out)) {
            cpu_set_last_error("concat: output must be contiguous");
            return C_FAILED;
        }

        // Get dtype from first input array
        int32_t dtype = in_arrays[0]->dtype;

        switch (dtype) {
        case INSIGHT_DTYPE_BOOL:
            concat_impl<bool>(in_arrays, (bool*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_U8:
            concat_impl<uint8_t>(in_arrays, (uint8_t*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_I8:
            concat_impl<int8_t>(in_arrays, (int8_t*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_I16:
            concat_impl<int16_t>(in_arrays, (int16_t*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_I32:
            concat_impl<int32_t>(in_arrays, (int32_t*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_I64:
            concat_impl<int64_t>(in_arrays, (int64_t*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_U16:
            concat_impl<uint16_t>(in_arrays, (uint16_t*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_U32:
            concat_impl<uint32_t>(in_arrays, (uint32_t*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_U64:
            concat_impl<uint64_t>(in_arrays, (uint64_t*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_F32:
            concat_impl<float>(in_arrays, (float*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_F64:
            concat_impl<double>(in_arrays, (double*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_C32:
            concat_impl<std::complex<float>>(in_arrays, (std::complex<float>*)out->data, ndim, out_dims, out_strides, axis);
            break;
        case INSIGHT_DTYPE_C64:
            concat_impl<std::complex<double>>(in_arrays, (std::complex<double>*)out->data, ndim, out_dims, out_strides, axis);
            break;
        default:
            cpu_set_last_error("concat: unsupported dtype");
            return C_FAILED;
        }

        return C_SUCCESS;
    }

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_BOOL, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_U8, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_I8, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_I16, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_I32, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_I64, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_U16, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_U32, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_U64, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_F32, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_F64, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_C32, concat_kernel_cpu);
REGISTER_CPU_KERNEL(concat, INSIGHT_DTYPE_C64, concat_kernel_cpu);