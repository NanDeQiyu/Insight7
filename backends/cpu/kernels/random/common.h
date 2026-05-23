// backends/cpu/kernels/random/common.h
/**
 * @file common.h
 * @brief Common utilities for CPU random kernels.
 * 
 * Provides thread-local RNG management and stride-aware iteration macros
 * for generating random numbers on arrays with arbitrary memory layouts.
 */

#ifndef CPU_RANDOM_COMMON_H
#define CPU_RANDOM_COMMON_H

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <random>
#include <atomic>
#include <thread>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Global Seed Management (CPU Backend)
// ============================================================================

/**
 * @brief Get the global seed for CPU backend.
 * 
 * @return uint64_t Current global seed
 */
static inline uint64_t cpu_get_global_seed(void) {
    static std::atomic<uint64_t> seed{0};
    return seed.load();
}

/**
 * @brief Set the global seed for CPU backend.
 * 
 * @param seed New global seed value
 */
static inline void cpu_set_global_seed(uint64_t seed) {
    static std::atomic<uint64_t> global_seed{0};
    global_seed.store(seed);
}

// ============================================================================
// Thread-Local RNG
// ============================================================================

/**
 * @brief Get thread-local RNG instance.
 * 
 * Each thread gets its own RNG seeded with global_seed + thread_id.
 * 
 * @return std::mt19937& Reference to thread-local RNG
 */
static inline std::mt19937& cpu_get_rng(void) {
    static thread_local std::mt19937 rng([]{
        uint64_t seed = cpu_get_global_seed();
        // Mix with thread ID for better distribution
        seed ^= static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        return static_cast<uint32_t>(seed);
    }());
    return rng;
}

// ============================================================================
// Stride Offset Calculation
// ============================================================================

/**
 * @brief Convert linear index to element offset using dimension strides.
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
// Random Distribution Macros
// ============================================================================

/**
 * @brief Macro for filling array with random values using a distribution.
 *
 * @param CTYPE    C data type (e.g., float, int32_t)
 * @param DIST     Distribution type (e.g., std::uniform_real_distribution<CTYPE>)
 * @param ARGS     Distribution constructor arguments
 */
#define RANDOM_FILL_LOOP(CTYPE, DIST, ARGS) \
    do { \
        CTYPE* out_data = (CTYPE*)out->data; \
        int64_t ndim = out->ndim; \
        int64_t dims[INSIGHT_MAX_NDIM]; \
        int64_t out_strides[INSIGHT_MAX_NDIM]; \
        for (int i = 0; i < ndim; ++i) { \
            dims[i] = out->dims[i]; \
            out_strides[i] = out->strides[i]; \
        } \
        int64_t n = out->numel; \
        _Pragma("omp parallel") \
        { \
            DIST dist ARGS; \
            _Pragma("omp for") \
            for (int64_t linear = 0; linear < n; ++linear) { \
                int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides); \
                out_data[off_out] = dist(cpu_get_rng()); \
            } \
        } \
    } while(0)

/**
 * @brief Macro for randperm (Fisher-Yates shuffle).
 */
#define RANDPERM_LOOP(CTYPE) \
    do { \
        CTYPE* out_data = (CTYPE*)out->data; \
        int64_t n = out->numel; \
        for (int64_t i = 0; i < n; ++i) { \
            out_data[i] = static_cast<CTYPE>(i); \
        } \
        for (int64_t i = n - 1; i > 0; --i) { \
            std::uniform_int_distribution<int64_t> dist(0, i); \
            int64_t j = dist(cpu_get_rng()); \
            CTYPE tmp = out_data[i]; \
            out_data[i] = out_data[j]; \
            out_data[j] = tmp; \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif // CPU_RANDOM_COMMON_H
