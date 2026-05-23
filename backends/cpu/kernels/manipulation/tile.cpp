// backends/cpu/kernels/manipulation/tile.cpp
/**
 * @file tile.cpp
 * @brief CPU kernel for tile operation.
 *
 * Repeats an array along each dimension.
 * Supports non-contiguous arrays via stride-based addressing.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (unused, outputs[0] is used instead)
 *   inputs[1] = InsightArray* input
 *   inputs[2] = InsightShape* reps
 *
 * @param inputs  See above
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include "insight/c_api/shape.h"
#include <complex>

 // ============================================================================
 // Tile helper template
 // ============================================================================

 /**
  * @brief Tile an array with stride support.
  *
  * @tparam T Data type
  * @param src_arr Source array (with possible offset and strides)
  * @param dst_arr Destination array (with possible offset and strides)
  * @param src_ndim Source dimensions
  * @param out_ndim Output dimensions
  * @param src_dims Source dimension sizes (right-aligned)
  * @param out_dims Output dimension sizes
  */
template<typename T>
static C_Status cpu_tile_impl(
    InsightArray* src_arr,
    InsightArray* dst_arr,
    int64_t src_ndim,
    int64_t out_ndim,
    const int64_t* src_dims,
    const int64_t* out_dims) {

    // Get source array properties
    const T* src = (const T*)src_arr->data;
    int64_t src_offset = src_arr->offset;
    int64_t src_strides[INSIGHT_MAX_NDIM];

    // Source strides (right-aligned)
    for (int64_t i = 0; i < src_ndim; ++i) {
        src_strides[out_ndim - src_ndim + i] = src_arr->strides[i];
    }

    // Get destination array properties
    T* dst = (T*)dst_arr->data;
    int64_t dst_offset = dst_arr->offset;
    int64_t dst_strides[INSIGHT_MAX_NDIM];
    for (int64_t i = 0; i < out_ndim; ++i) {
        dst_strides[i] = dst_arr->strides[i];
    }

    int64_t total = 1;
    for (int64_t d = 0; d < out_ndim; ++d) {
        total *= out_dims[d];
    }

    int64_t offset = out_ndim - src_ndim;

#pragma omp parallel for
    for (int64_t linear = 0; linear < total; ++linear) {
        int64_t indices[INSIGHT_MAX_NDIM];
        cpu_linear_to_indices(linear, out_ndim, out_dims, indices);

        // Compute source offset (mod by source dimensions)
        int64_t src_off = src_offset;
        for (int64_t d = 0; d < src_ndim; ++d) {
            int64_t idx = indices[offset + d] % src_dims[d];
            src_off += idx * src_strides[offset + d];
        }

        // Compute destination offset
        int64_t dst_off = dst_offset;
        for (int64_t d = 0; d < out_ndim; ++d) {
            dst_off += indices[d] * dst_strides[d];
        }

        dst[dst_off] = src[src_off];
    }

    return C_SUCCESS;
}

// ============================================================================
// Type dispatch macro
// ============================================================================

#define TILE_CASE(DTYPE, CTYPE) \
    case DTYPE: \
        return cpu_tile_impl<CTYPE>(x, out, src_ndim, out_ndim, src_dims, out_dims);

// ============================================================================
// Kernel entry point
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

    C_Status tile_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* out = (InsightArray*)outputs[0];
        InsightArray* x = (InsightArray*)inputs[1];

        if (!out || !x) {
            cpu_set_last_error("tile: null array pointer");
            return C_FAILED;
        }

        if (!inputs[2]) {
            cpu_set_last_error("tile: reps is null");
            return C_FAILED;
        }

        InsightShape* reps = (InsightShape*)inputs[2];

        int64_t src_ndim = x->ndim;
        int64_t out_ndim = out->ndim;

        // Source dimensions (right-aligned)
        int64_t src_dims[INSIGHT_MAX_NDIM] = { 0 };
        for (int64_t i = 0; i < src_ndim; ++i) {
            src_dims[out_ndim - src_ndim + i] = x->dims[i];
        }

        // Output dimensions
        int64_t out_dims[INSIGHT_MAX_NDIM];
        for (int64_t i = 0; i < out_ndim; ++i) {
            out_dims[i] = out->dims[i];
        }

        switch (x->dtype) {
            TILE_CASE(INSIGHT_DTYPE_BOOL, bool)
                TILE_CASE(INSIGHT_DTYPE_U8, uint8_t)
                TILE_CASE(INSIGHT_DTYPE_I8, int8_t)
                TILE_CASE(INSIGHT_DTYPE_I16, int16_t)
                TILE_CASE(INSIGHT_DTYPE_I32, int32_t)
                TILE_CASE(INSIGHT_DTYPE_I64, int64_t)
                TILE_CASE(INSIGHT_DTYPE_U16, uint16_t)
                TILE_CASE(INSIGHT_DTYPE_U32, uint32_t)
                TILE_CASE(INSIGHT_DTYPE_U64, uint64_t)
                TILE_CASE(INSIGHT_DTYPE_F16, uint16_t)
                TILE_CASE(INSIGHT_DTYPE_BF16, uint16_t)
                TILE_CASE(INSIGHT_DTYPE_F32, float)
                TILE_CASE(INSIGHT_DTYPE_F64, double)
                TILE_CASE(INSIGHT_DTYPE_C32, std::complex<float>)
                TILE_CASE(INSIGHT_DTYPE_C64, std::complex<double>)
                TILE_CASE(INSIGHT_DTYPE_F8_E4M3, uint8_t)
                TILE_CASE(INSIGHT_DTYPE_F8_E5M2, uint8_t)
        default:
            cpu_set_last_error("tile: unsupported dtype");
            return C_FAILED;
        }
    }

#ifdef __cplusplus
}
#endif

// ============================================================================
// Kernel registration for all supported types
// ============================================================================

REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_BOOL, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_U8, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_I8, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_I16, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_I32, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_I64, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_U16, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_U32, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_U64, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_F16, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_BF16, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_F32, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_F64, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_C32, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_C64, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_F8_E4M3, tile_kernel_cpu);
REGISTER_CPU_KERNEL(tile, INSIGHT_DTYPE_F8_E5M2, tile_kernel_cpu);