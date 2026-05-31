// backends/cpu/kernels/fft/common.h
/**
 * @file common.h
 * @brief Common utilities for CPU FFT kernels.
 *
 * Provides thread-local FFTW plan caching shared across all FFT kernels.
 */

#ifndef CPU_FFT_COMMON_H
#define CPU_FFT_COMMON_H

#ifdef INSIGHT_USE_FFTW3

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <fftw3.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// FFT Plan Kind Enumeration
// ============================================================================

/**
 * @brief Type of FFT transform.
 */
typedef enum {
  FFT_KIND_C2C = 0, ///< Complex to complex
  FFT_KIND_R2C = 1, ///< Real to complex
  FFT_KIND_C2R = 2  ///< Complex to real
} FFTKind;

// ============================================================================
// Thread-Local Plan Cache (Shared Across All FFT Kernels)
// ============================================================================

/**
 * @brief Global FFT plan cache structure.
 *
 * Shared by all FFT kernels (fft, rfft, irfft) within the same thread.
 * Each thread has its own cache instance.
 */
typedef struct {
  fftw_plan plan_f64;  ///< Double precision plan
  fftwf_plan plan_f32; ///< Single precision plan
  void *buf_f64;       ///< Buffer used to create f64 plan
  void *buf_f32;       ///< Buffer used to create f32 plan
  int64_t n;           ///< FFT length
  int64_t batch;       ///< Number of transforms
  int direction;       ///< FFTW_FORWARD or FFTW_BACKWARD
  FFTKind kind;        ///< Type of transform (C2C, R2C, C2R)
} FFTWCache;

/**
 * @brief Get thread-local FFT cache.
 *
 * @return Pointer to thread-local cache
 */
FFTWCache *fft_get_cache(void);

/**
 * @brief Invalidate the current cache (force plan recreation).
 */
static inline void fft_invalidate_cache(void) {
  FFTWCache *cache = fft_get_cache();
  if (cache->plan_f64) {
    fftw_destroy_plan(cache->plan_f64);
    cache->plan_f64 = NULL;
  }
  if (cache->plan_f32) {
    fftwf_destroy_plan(cache->plan_f32);
    cache->plan_f32 = NULL;
  }
  cache->buf_f64 = NULL;
  cache->buf_f32 = NULL;
  cache->n = 0;
  cache->batch = 0;
  cache->direction = 0;
  cache->kind = FFT_KIND_C2C;
}

/**
 * @brief Ensure FFTW plan exists for given parameters (double precision).
 *
 * For C2C transforms, creates plan with the actual buffer (required for
 * correctness at n>=64). For R2C/C2R, uses dummy buffers (faster).
 *
 * @param n        FFT length
 * @param batch    Number of transforms
 * @param direction FFTW_FORWARD or FFTW_BACKWARD
 * @param kind     Transform kind (C2C, R2C, C2R)
 * @param buf      Actual data buffer (used for C2C plan creation)
 * @return fftw_plan Valid plan
 */
fftw_plan fft_ensure_plan_f64(int n, int64_t batch, int direction, FFTKind kind,
                              void *buf);

/**
 * @brief Ensure FFTW plan exists for given parameters (single precision).
 */
fftwf_plan fft_ensure_plan_f32(int n, int64_t batch, int direction,
                               FFTKind kind, void *buf);

#ifdef __cplusplus
}
#endif

#endif // INSIGHT_USE_FFTW3

#endif // CPU_FFT_COMMON_H