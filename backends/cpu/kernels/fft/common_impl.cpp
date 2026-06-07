// backends/cpu/kernels/fft/common_impl.cpp
/**
 * @file common_impl.cpp
 * @brief FFTW plan cache implementation.
 *
 * - C2C: always recreate with actual buffers (FFTW_ESTIMATE buffer-address
 *   dependency for n >= 64 — using dummy buffers gives wrong results).
 * - R2C/C2R: multi-plan cache with dummy buffers (works correctly).
 */

#ifdef INSIGHT_USE_FFTW3

#include "common.h"
#include <cstring>

static thread_local FFTWCache cache = {{0}};

FFTWCache *fft_get_cache(void) { return &cache; }

fftw_plan fft_ensure_plan_f64(int n, int64_t batch, int direction, FFTKind kind,
                              void *buf_in, void *buf_out) {
  if (kind == FFT_KIND_C2C) {
    // C2C must use actual buffers
    int n_int = (int)n;
    return fftw_plan_many_dft(1, &n_int, (int)batch, (fftw_complex *)buf_in,
                              NULL, 1, n_int, (fftw_complex *)buf_out, NULL, 1,
                              n_int, direction, FFTW_ESTIMATE);
  }
  // R2C/C2R: use multi-plan cache (dummy buffers work correctly)
  return (fftw_plan)fft_cache_find_or_create(n, batch, direction, kind, 1);
}

fftwf_plan fft_ensure_plan_f32(int n, int64_t batch, int direction,
                               FFTKind kind, void *buf_in, void *buf_out) {
  if (kind == FFT_KIND_C2C) {
    // C2C must use actual buffers
    int n_int = (int)n;
    return fftwf_plan_many_dft(1, &n_int, (int)batch, (fftwf_complex *)buf_in,
                               NULL, 1, n_int, (fftwf_complex *)buf_out, NULL,
                               1, n_int, direction, FFTW_ESTIMATE);
  }
  // R2C/C2R: use multi-plan cache (dummy buffers work correctly)
  return (fftwf_plan)fft_cache_find_or_create(n, batch, direction, kind, 0);
}

#endif // INSIGHT_USE_FFTW3
