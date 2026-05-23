// backends/cpu/kernels/elementwise/common.h
/**
 * @file common.h
 * @brief Common utilities for CPU elementwise kernels.
 * 
 * This header provides stride-aware iteration macros and helper functions
 * for implementing elementwise operations on arrays with arbitrary memory
 * layouts (contiguous or strided).
 * 
 * All kernels that include this header must be compiled with OpenMP support
 * for parallel loop execution.
 */

#ifndef CPU_ELEMENTWISE_COMMON_H
#define CPU_ELEMENTWISE_COMMON_H

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>

// ============================================================================
// Stride Offset Calculation
// ============================================================================

/**
 * @brief Convert linear index to element offset using dimension strides.
 * 
 * Given a linear index in row-major order, compute the corresponding
 * memory offset for an array with given dimensions and strides.
 * 
 * @param linear   Linear index (0 .. total_elements-1)
 * @param ndim     Number of dimensions
 * @param dims     Array of dimension sizes (length ndim)
 * @param strides  Array of strides in elements (length ndim)
 * @return Memory offset in elements
 */
static inline int64_t cpu_offset_from_linear(
    int64_t linear,
    int64_t ndim,
    const int64_t* dims,
    const int64_t* strides) {
    
    int64_t offset = 0;
    int64_t remaining = linear;
    
    for (int64_t d = ndim - 1; d >= 0; --d) {
        int64_t idx = remaining % dims[d];
        remaining /= dims[d];
        offset += idx * strides[d];
    }
    
    return offset;
}
// ============================================================================
// Elementwise Operation Macro
// ============================================================================

/**
 * @brief Macro for elementwise binary operations with stride support.
 *
 * This macro expands to a complete loop that iterates over all elements
 * of the output array, computes offsets for input and output arrays using
 * their respective strides and offsets, and applies the given operation.
 *
 * @param CTYPE    C data type (e.g., float, int32_t, std::complex<float>)
 * @param OP       Operation expression using 'a' and 'b' as operands
 */
#define ELEMENTWISE_KERNEL_LOOP(CTYPE, OP) \
    do { \
        CTYPE* a_data = (CTYPE*)a->data; \
        CTYPE* b_data = (CTYPE*)b->data; \
        CTYPE* out_data = (CTYPE*)out->data; \
        int64_t ndim = out->ndim; \
        int64_t dims[INSIGHT_MAX_NDIM]; \
        int64_t a_strides[INSIGHT_MAX_NDIM]; \
        int64_t b_strides[INSIGHT_MAX_NDIM]; \
        int64_t out_strides[INSIGHT_MAX_NDIM]; \
        for (int i = 0; i < ndim; ++i) { \
            dims[i] = out->dims[i]; \
            a_strides[i] = a->strides[i]; \
            b_strides[i] = b->strides[i]; \
            out_strides[i] = out->strides[i]; \
        } \
        int64_t n = out->numel; \
        _Pragma("omp parallel for") \
        for (int64_t linear = 0; linear < n; ++linear) { \
            int64_t off_a = a->offset + cpu_offset_from_linear(linear, ndim, dims, a_strides); \
            int64_t off_b = b->offset + cpu_offset_from_linear(linear, ndim, dims, b_strides); \
            int64_t off_out = out->offset + cpu_offset_from_linear(linear, ndim, dims, out_strides); \
            out_data[off_out] = a_data[off_a] OP b_data[off_b]; \
        } \
    } while(0)

 /**
  * @brief Macro for elementwise binary operations with stride support.
  *
  * This macro expands to a complete loop that iterates over all elements
  * of the output array, computes offsets for input and output arrays using
  * their respective strides, and applies the given operation.
  *
  * @param CTYPE    C data type (e.g., float, int32_t, std::complex<float>)
  * @param Func     Function to apply to each pair of elements
  */
#define ELEMENTWISE_KERNEL_LOOP_Func(CTYPE, Func) \
    do { \
        CTYPE* a_data = (CTYPE*)a->data; \
        CTYPE* b_data = (CTYPE*)b->data; \
        CTYPE* out_data = (CTYPE*)out->data; \
        int64_t ndim = out->ndim; \
        int64_t dims[INSIGHT_MAX_NDIM]; \
        int64_t a_strides[INSIGHT_MAX_NDIM]; \
        int64_t b_strides[INSIGHT_MAX_NDIM]; \
        int64_t out_strides[INSIGHT_MAX_NDIM]; \
        for (int i = 0; i < ndim; ++i) { \
            dims[i] = out->dims[i]; \
            a_strides[i] = a->strides[i]; \
            b_strides[i] = b->strides[i]; \
            out_strides[i] = out->strides[i]; \
        } \
        int64_t n = out->numel; \
        _Pragma("omp parallel for") \
        for (int64_t linear = 0; linear < n; ++linear) { \
            int64_t off_a = a->offset + cpu_offset_from_linear(linear, ndim, dims, a_strides); \
            int64_t off_b = b->offset + cpu_offset_from_linear(linear, ndim, dims, b_strides); \
            int64_t off_out = out->offset + cpu_offset_from_linear(linear, ndim, dims, out_strides); \
            out_data[off_out] = Func(a_data[off_a], b_data[off_b]); \
        } \
    } while(0)

  // ============================================================================
  // Comparison Operation Macro (returns bool)
  // ============================================================================

  /**
   * @brief Macro for elementwise comparison operations with stride support.
   *
   * Similar to ELEMENTWISE_KERNEL_LOOP but output type is bool.
   *
   * @param CTYPE    C data type of input arrays
   * @param OP       Comparison expression returning bool
   */
#define ELEMENTWISE_CMP_LOOP(CTYPE, OP) \
    do { \
        CTYPE* a_data = (CTYPE*)a->data; \
        CTYPE* b_data = (CTYPE*)b->data; \
        bool* out_data = (bool*)out->data; \
        int64_t ndim = out->ndim; \
        int64_t dims[INSIGHT_MAX_NDIM]; \
        int64_t a_strides[INSIGHT_MAX_NDIM]; \
        int64_t b_strides[INSIGHT_MAX_NDIM]; \
        int64_t out_strides[INSIGHT_MAX_NDIM]; \
        for (int i = 0; i < ndim; ++i) { \
            dims[i] = out->dims[i]; \
            a_strides[i] = a->strides[i]; \
            b_strides[i] = b->strides[i]; \
            out_strides[i] = out->strides[i]; \
        } \
        int64_t n = out->numel; \
        _Pragma("omp parallel for") \
        for (int64_t linear = 0; linear < n; ++linear) { \
            int64_t off_a = a->offset + cpu_offset_from_linear(linear, ndim, dims, a_strides); \
            int64_t off_b = b->offset + cpu_offset_from_linear(linear, ndim, dims, b_strides); \
            int64_t off_out = out->offset + cpu_offset_from_linear(linear, ndim, dims, out_strides); \
            out_data[off_out] = a_data[off_a] OP b_data[off_b]; \
        } \
    } while(0)

   /**
    * @brief Macro for elementwise comparison operations with stride support.
    *
    * Similar to ELEMENTWISE_KERNEL_LOOP but output type is bool.
    *
    * @param CTYPE    C data type of input arrays
    * @param Func     Function returning bool for comparison
    */
#define ELEMENTWISE_CMP_LOOP_Func(CTYPE, Func) \
    do { \
        CTYPE* a_data = (CTYPE*)a->data; \
        CTYPE* b_data = (CTYPE*)b->data; \
        bool* out_data = (bool*)out->data; \
        int64_t ndim = out->ndim; \
        int64_t dims[INSIGHT_MAX_NDIM]; \
        int64_t a_strides[INSIGHT_MAX_NDIM]; \
        int64_t b_strides[INSIGHT_MAX_NDIM]; \
        int64_t out_strides[INSIGHT_MAX_NDIM]; \
        for (int i = 0; i < ndim; ++i) { \
            dims[i] = out->dims[i]; \
            a_strides[i] = a->strides[i]; \
            b_strides[i] = b->strides[i]; \
            out_strides[i] = out->strides[i]; \
        } \
        int64_t n = out->numel; \
        _Pragma("omp parallel for") \
        for (int64_t linear = 0; linear < n; ++linear) { \
            int64_t off_a = a->offset + cpu_offset_from_linear(linear, ndim, dims, a_strides); \
            int64_t off_b = b->offset + cpu_offset_from_linear(linear, ndim, dims, b_strides); \
            int64_t off_out = out->offset + cpu_offset_from_linear(linear, ndim, dims, out_strides); \
            out_data[off_out] = Func(a_data[off_a], b_data[off_b]); \
        } \
    } while(0)
#endif // CPU_ELEMENTWISE_COMMON_H
