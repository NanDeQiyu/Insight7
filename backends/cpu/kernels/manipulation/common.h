// backends/cpu/kernels/manipulation/common.h
/**
 * @file common.h
 * @brief Common utilities for CPU manipulation kernels.
 * 
 * Provides helper functions and macros for manipulation operations.
 */

#ifndef CPU_MANIPULATION_COMMON_H
#define CPU_MANIPULATION_COMMON_H

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Helper: Convert linear index to multi-dimensional indices
// ============================================================================

/**
 * @brief Convert linear index to multi-dimensional indices.
 * 
 * @param linear Linear index
 * @param ndim Number of dimensions
 * @param dims Dimension sizes
 * @param indices Output array of indices (size ndim)
 */
static inline void cpu_linear_to_indices(
    int64_t linear,
    int64_t ndim,
    const int64_t* dims,
    int64_t* indices) {
    
    int64_t remaining = linear;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        indices[d] = remaining % dims[d];
        remaining /= dims[d];
    }
}

/**
 * @brief Convert multi-dimensional indices to linear index.
 * 
 * @param ndim Number of dimensions
 * @param dims Dimension sizes
 * @param strides Strides in elements
 * @param indices Multi-dimensional indices
 * @return Linear index
 */
static inline int64_t cpu_indices_to_linear(
    int64_t ndim,
    const int64_t* dims,
    const int64_t* strides,
    const int64_t* indices) {
    
    int64_t linear = 0;
    for (int64_t d = 0; d < ndim; ++d) {
        linear += indices[d] * strides[d];
    }
    return linear;
}

/**
 * @brief Compute output strides for a given shape (row-major).
 * 
 * @param ndim Number of dimensions
 * @param dims Dimension sizes
 * @param strides Output strides (size ndim)
 */
static inline void cpu_compute_strides(
    int64_t ndim,
    const int64_t* dims,
    int64_t* strides) {
    
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int64_t d = ndim - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * dims[d + 1];
    }
}

#endif // CPU_MANIPULATION_COMMON_H
