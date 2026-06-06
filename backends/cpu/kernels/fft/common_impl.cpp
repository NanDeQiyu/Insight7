/**
 * @file common_impl.cpp
 * @brief FFTW plan cache implementation.
 *
 * - C2C: plans created with actual buffer each time (FFTW_ESTIMATE +
 *   fftw_execute). fftw_execute_dft gives wrong results for n>=64 with
 *   FFTW_ESTIMATE plans created on different buffers.
 * - R2C/C2R: plans cached via fft_cache_find_or_create() with dummy buffers
 *   (fftw_execute_dft_r2c/c2r works correctly with any buffer).
 */

#ifdef INSIGHT_USE_FFTW3

#include "common.h"
#include <cstring>

static thread_local FFTWCache cache = {{{0}}};

FFTWCache *fft_get_cache(void) { return &cache; }

fftw_plan fft_ensure_plan_f64(int n, int64_t batch, int direction, FFTKind kind,
                              void *buf) {
  // C2C: always recreate with actual buffer (FFTW_ESTIMATE buffer dependency)
  if (kind == FFT_KIND_C2C) {
    int n_int = (int)n;
    return fftw_plan_many_dft(1, &n_int, (int)batch, (fftw_complex *)buf, NULL,
                              1, n_int, (fftw_complex *)buf, NULL, 1, n_int,
                              direction, FFTW_ESTIMATE);
  }

  // R2C/C2R: use multi-plan cache
  return (fftw_plan)fft_cache_find_or_create(n, batch, direction, kind, 1);
}

fftwf_plan fft_ensure_plan_f32(int n, int64_t batch, int direction,
                               FFTKind kind, void *buf) {
  // C2C: always recreate with actual buffer (FFTW_ESTIMATE buffer dependency)
  if (kind == FFT_KIND_C2C) {
    int n_int = (int)n;
    return fftwf_plan_many_dft(1, &n_int, (int)batch, (fftwf_complex *)buf,
                               NULL, 1, n_int, (fftwf_complex *)buf, NULL, 1,
                               n_int, direction, FFTW_ESTIMATE);
  }

  // R2C/C2R: use multi-plan cache
  return (fftwf_plan)fft_cache_find_or_create(n, batch, direction, kind, 0);
}

#endif // INSIGHT_USE_FFTW3
