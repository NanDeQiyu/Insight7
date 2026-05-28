// backends/cpu/kernels/indexing/put_along_axis.cpp
/**
 * @file put_along_axis.cpp
 * @brief CPU kernel for put_along_axis operation.
 *
 * Puts values into an array using indices array along an axis.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (will be modified)
 *   inputs[1] = InsightArray* indices (original shape, not broadcasted)
 *   inputs[2] = InsightArray* values
 *   inputs[3] = int* axis
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <complex>
#include <cstring>

template <typename T>
static void
put_along_axis_impl(T* dst, const int64_t* idx_data, const T* val_data,
	int64_t total_indices, int64_t out_ndim,
	const int64_t* out_dims, const int64_t* out_strides,
	int64_t idx_ndim, const int64_t* idx_dims,
	const int64_t* idx_strides, int axis, int64_t val_size) {

	// Calculate the coordinates of idx
	int64_t coord[INSIGHT_MAX_NDIM];

	for (int64_t linear = 0; linear < total_indices; ++linear) {
		// Decode coordinates with shape of idx
		int64_t tmp = linear;
		for (int d = idx_ndim - 1; d >= 0; --d) {
			coord[d] = tmp % idx_dims[d];
			tmp /= idx_dims[d];
		}

		// Get the position from indices
		int64_t pos = idx_data[linear];
		if (pos < 0)
			pos += out_dims[axis];

		// Calculate the output offset: replace the coordinates on axis with pos, and use the values ​​in coord for other dimensions.
		// Note: coord is the coordinate of idx. The shape of idx may be smaller than out. Non-axis dimensions need to be broadcast.
		int64_t out_offset = 0;
		for (int d = 0; d < out_ndim; ++d) {
			if (d == axis) {
				out_offset += pos * out_strides[d];
			}
			else {
				// For non-axis dimensions, if idx has this dimension, use the value in coord, otherwise use 0
				int64_t c = 0;
				if (d < idx_ndim) {
					c = coord[d];
				}
				out_offset += c * out_strides[d];
			}
		}

		dst[out_offset] = val_data[linear % val_size];
	}
}

#ifdef __cplusplus
extern "C" {
#endif

	C_Status put_along_axis_kernel_cpu(void** inputs, void** outputs) {
		InsightArray* out = (InsightArray*)outputs[0];
		InsightArray* idx = (InsightArray*)inputs[1];
		InsightArray* val = (InsightArray*)inputs[2];
		int axis = *(int*)inputs[3];

		if (!out || !idx || !val) {
			cpu_set_last_error("put_along_axis: null array pointer");
			return C_FAILED;
		}

		int64_t out_ndim = out->ndim;
		int64_t out_dims[INSIGHT_MAX_NDIM];
		int64_t out_strides[INSIGHT_MAX_NDIM];
		for (int i = 0; i < out_ndim; ++i) {
			out_dims[i] = out->dims[i];
			out_strides[i] = out->strides[i];
		}

		int64_t idx_ndim = idx->ndim;
		int64_t idx_dims[INSIGHT_MAX_NDIM];
		int64_t idx_strides[INSIGHT_MAX_NDIM];
		for (int i = 0; i < idx_ndim; ++i) {
			idx_dims[i] = idx->dims[i];
			idx_strides[i] = idx->strides[i];
		}

		int64_t total_indices = idx->numel;
		int64_t* idx_data = (int64_t*)idx->data;
		int64_t val_size = val->numel;

		// Boundary checking
		for (int64_t i = 0; i < total_indices; ++i) {
			int64_t pos = idx_data[i];
			if (pos < 0)
				pos += out_dims[axis];
			if (pos < 0 || pos >= out_dims[axis]) {
				cpu_set_last_error("put_along_axis: index out of bounds");
				return C_FAILED;
			}
		}

		switch (out->dtype) {
		case INSIGHT_DTYPE_BOOL:
			put_along_axis_impl<bool>((bool*)out->data, idx_data,
				(const bool*)val->data, total_indices, out_ndim,
				out_dims, out_strides, idx_ndim, idx_dims,
				idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_U8:
			put_along_axis_impl<uint8_t>((uint8_t*)out->data, idx_data,
				(const uint8_t*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_I8:
			put_along_axis_impl<int8_t>((int8_t*)out->data, idx_data,
				(const int8_t*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_I16:
			put_along_axis_impl<int16_t>((int16_t*)out->data, idx_data,
				(const int16_t*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_I32:
			put_along_axis_impl<int32_t>((int32_t*)out->data, idx_data,
				(const int32_t*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_I64:
			put_along_axis_impl<int64_t>((int64_t*)out->data, idx_data,
				(const int64_t*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_U16:
			put_along_axis_impl<uint16_t>((uint16_t*)out->data, idx_data,
				(const uint16_t*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_U32:
			put_along_axis_impl<uint32_t>((uint32_t*)out->data, idx_data,
				(const uint32_t*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_U64:
			put_along_axis_impl<uint64_t>((uint64_t*)out->data, idx_data,
				(const uint64_t*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_F16:
		case INSIGHT_DTYPE_BF16:
			put_along_axis_impl<uint16_t>((uint16_t*)out->data, idx_data,
				(const uint16_t*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_F32:
			put_along_axis_impl<float>((float*)out->data, idx_data,
				(const float*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_F64:
			put_along_axis_impl<double>((double*)out->data, idx_data,
				(const double*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_C32:
			put_along_axis_impl<std::complex<float>>(
				(std::complex<float> *)out->data, idx_data,
				(const std::complex<float> *)val->data, total_indices, out_ndim,
				out_dims, out_strides, idx_ndim, idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_C64:
			put_along_axis_impl<std::complex<double>>(
				(std::complex<double> *)out->data, idx_data,
				(const std::complex<double> *)val->data, total_indices, out_ndim,
				out_dims, out_strides, idx_ndim, idx_dims, idx_strides, axis, val_size);
			break;
		case INSIGHT_DTYPE_F8_E4M3:
		case INSIGHT_DTYPE_F8_E5M2:
			put_along_axis_impl<uint8_t>((uint8_t*)out->data, idx_data,
				(const uint8_t*)val->data, total_indices,
				out_ndim, out_dims, out_strides, idx_ndim,
				idx_dims, idx_strides, axis, val_size);
			break;
		default:
			cpu_set_last_error("put_along_axis: unsupported dtype");
			return C_FAILED;
		}

		return C_SUCCESS;
	}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_BOOL,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_U8,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_I8,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_I16,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_I32,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_I64,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_U16,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_U32,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_U64,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_F16,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_BF16,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_F32,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_F64,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_C32,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_C64,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_F8_E4M3,
	put_along_axis_kernel_cpu);
REGISTER_CPU_KERNEL(put_along_axis, INSIGHT_DTYPE_F8_E5M2,
	put_along_axis_kernel_cpu);