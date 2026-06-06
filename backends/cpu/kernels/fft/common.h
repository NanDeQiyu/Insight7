/**
 * @file common.h
 * @brief Common utilities for CPU FFT kernels.
 *
 * Provides thread-local FFTW plan caching shared across all FFT kernels.
 *
 * Cache strategy:
 * - C2C: Always recreates plan with actual buffer (FFTW_ESTIMATE has
 *   buffer-address dependency for n>=64 — fftw_execute_dft gives wrong
 *   results with mismatched buffer). Plan creation with FFTW_ESTIMATE
 *   is fast (~microseconds), so this is acceptable.
 * - R2C/C2R: Multi-plan cache keyed by (n, batch, direction, kind, is_f64).
 *   These transforms use fftw_execute_dft_r2c/c2r which work correctly
 *   with any buffer, so plans can be cached with dummy buffers.
 */

#ifndef CPU_FFT_COMMON_H
#define CPU_FFT_COMMON_H

#ifdef INSIGHT_USE_FFTW3

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <fftw3.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
// Multi-Plan Cache Entry (for R2C/C2R only)
// ============================================================================

#define FFTW_CACHE_MAX_PLANS 8

/**
 * @brief A single cache entry for a cached FFTW plan.
 */
typedef struct {
  int64_t n;
  int64_t batch;
  int direction;       ///< FFTW_FORWARD or FFTW_BACKWARD
  FFTKind kind;        ///< R2C or C2R only
  int is_f64;          ///< 1 for f64, 0 for f32
  fftw_plan plan_f64;  ///< Double precision plan (only if is_f64)
  fftwf_plan plan_f32; ///< Single precision plan (only if !is_f64)
  int valid;           ///< 1 if entry is occupied
} FFTWCacheEntry;

/**
 * @brief Thread-local FFTW plan cache.
 *
 * - C2C plans are NOT cached (always recreated with actual buffer).
 * - R2C/C2R plans are cached in entries[].
 */
typedef struct {
  FFTWCacheEntry entries[FFTW_CACHE_MAX_PLANS];
  int entry_count; ///< Number of occupied entries
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
  for (int i = 0; i < cache->entry_count; ++i) {
    if (cache->entries[i].valid) {
      if (cache->entries[i].is_f64) {
        fftw_destroy_plan(cache->entries[i].plan_f64);
      } else {
        fftwf_destroy_plan(cache->entries[i].plan_f32);
      }
    }
  }
  memset(cache->entries, 0, sizeof(cache->entries));
  cache->entry_count = 0;
}

/**
 * @brief Find or create a cached FFTW plan for R2C/C2R.
 *
 * For C2C transforms, this always returns NULL — the caller must create
 * a plan with the actual buffer.
 *
 * @param n         FFT length
 * @param batch     Number of transforms
 * @param direction FFTW_FORWARD or FFTW_BACKWARD
 * @param kind      Transform kind (R2C or C2R)
 * @param is_f64    1 for double precision, 0 for single
 * @return void* Plan pointer (fftw_plan or fftwf_plan), NULL for C2C or error
 */
static inline void *fft_cache_find_or_create(int n, int64_t batch,
                                             int direction, FFTKind kind,
                                             int is_f64) {
  if (kind == FFT_KIND_C2C)
    return NULL; // C2C must be created with actual buffer

  FFTWCache *cache = fft_get_cache();

  // Search existing entries
  for (int i = 0; i < cache->entry_count; ++i) {
    FFTWCacheEntry *e = &cache->entries[i];
    if (e->valid && e->n == n && e->batch == batch &&
        e->direction == direction && e->kind == kind && e->is_f64 == is_f64) {
      return is_f64 ? (void *)e->plan_f64 : (void *)e->plan_f32;
    }
  }

  // Evict oldest if full
  int idx = cache->entry_count;
  if (idx >= FFTW_CACHE_MAX_PLANS) {
    // Evict entry 0, shift remaining
    if (cache->entries[0].valid) {
      if (cache->entries[0].is_f64) {
        fftw_destroy_plan(cache->entries[0].plan_f64);
      } else {
        fftwf_destroy_plan(cache->entries[0].plan_f32);
      }
    }
    memmove(&cache->entries[0], &cache->entries[1],
            (FFTW_CACHE_MAX_PLANS - 1) * sizeof(FFTWCacheEntry));
    idx = FFTW_CACHE_MAX_PLANS - 1;
    cache->entries[idx].valid = 0;
  } else {
    cache->entry_count = idx + 1;
  }

  FFTWCacheEntry *e = &cache->entries[idx];
  int n_int = n;

  // Create plan with dummy buffers
  if (kind == FFT_KIND_R2C || kind == FFT_KIND_C2R) {
    if (is_f64) {
      double *di = (double *)fftw_malloc((size_t)n * sizeof(double));
      double *dout =
          (double *)fftw_malloc((size_t)(n / 2 + 1) * 2 * sizeof(double));
      if (kind == FFT_KIND_R2C) {
        e->plan_f64 = fftw_plan_many_dft_r2c(1, &n_int, (int)batch, di, NULL, 1,
                                             n_int, (fftw_complex *)dout, NULL,
                                             1, n / 2 + 1, FFTW_ESTIMATE);
      } else {
        e->plan_f64 = fftw_plan_many_dft_c2r(
            1, &n_int, (int)batch, (fftw_complex *)di, NULL, 1, n / 2 + 1, dout,
            NULL, 1, n_int, FFTW_ESTIMATE);
      }
      fftw_free(di);
      fftw_free(dout);
    } else {
      float *di = (float *)fftwf_malloc((size_t)n * sizeof(float));
      float *dout =
          (float *)fftwf_malloc((size_t)(n / 2 + 1) * 2 * sizeof(float));
      if (kind == FFT_KIND_R2C) {
        e->plan_f32 = fftwf_plan_many_dft_r2c(
            1, &n_int, (int)batch, di, NULL, 1, n_int, (fftwf_complex *)dout,
            NULL, 1, n / 2 + 1, FFTW_ESTIMATE);
      } else {
        e->plan_f32 = fftwf_plan_many_dft_c2r(
            1, &n_int, (int)batch, (fftwf_complex *)di, NULL, 1, n / 2 + 1,
            dout, NULL, 1, n_int, FFTW_ESTIMATE);
      }
      fftwf_free(di);
      fftwf_free(dout);
    }
  }

  e->n = n;
  e->batch = batch;
  e->direction = direction;
  e->kind = kind;
  e->is_f64 = is_f64;
  e->valid = 1;

  return is_f64 ? (void *)e->plan_f64 : (void *)e->plan_f32;
}

/**
 * @brief Ensure FFTW plan exists for given parameters (double precision).
 *
 * For C2C transforms, creates plan with the actual buffer (required for
 * correctness at n>=64). For R2C/C2R, uses the multi-plan cache.
 *
 * @param n        FFT length
 * @param batch    Number of transforms
 * @param direction FFTW_FORWARD or FFTW_BACKWARD
 * @param kind     Transform kind (C2C, R2C, C2R)
 * @param buf_in   Input buffer (for C2C, same as buf_out)
 * @param buf_out  Output buffer (for C2C, same as buf_in)
 * @return fftw_plan Valid plan
 */
fftw_plan fft_ensure_plan_f64(int n, int64_t batch, int direction, FFTKind kind,
                              void *buf_in, void *buf_out);

/**
 * @brief Ensure FFTW plan exists for given parameters (single precision).
 */
fftwf_plan fft_ensure_plan_f32(int n, int64_t batch, int direction,
                               FFTKind kind, void *buf_in, void *buf_out);

#ifdef __cplusplus
}
#endif

#endif // INSIGHT_USE_FFTW3

#endif // CPU_FFT_COMMON_H
