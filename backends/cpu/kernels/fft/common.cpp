// backends/cpu/kernels/fft/common.c
/**
 * @file common.c
 * @brief Implementation of shared FFT plan cache.
 */
#ifdef INSIGHT_USE_FFTW3

#include "common.h"

// Thread-local cache instance (shared across all kernel files in this thread)
static thread_local FFTWCache g_fft_cache = {
    NULL, NULL, FFT_KIND_C2C, FFT_KIND_C2C, FFT_KIND_C2C, FFT_KIND_C2C};

extern "C" {

FFTWCache *fft_get_cache(void) { return &g_fft_cache; }

fftw_plan fft_ensure_plan_f64(int n, int64_t batch, int direction,
                              FFTKind kind) {
  FFTWCache *cache = fft_get_cache();

  if (cache->plan_f64 && cache->n == n && cache->batch == batch &&
      cache->direction == direction && cache->kind == kind) {
    return cache->plan_f64;
  }

  // Destroy old plan
  if (cache->plan_f64) {
    fftw_destroy_plan(cache->plan_f64);
  }

  // Create new plan
  if (kind == FFT_KIND_R2C) {
    double *dummy_in = (double *)fftw_malloc(n * batch * sizeof(double));
    fftw_complex *dummy_out =
        (fftw_complex *)fftw_malloc((n / 2 + 1) * batch * sizeof(fftw_complex));
    cache->plan_f64 =
        fftw_plan_many_dft_r2c(1, &n, batch, dummy_in, NULL, 1, n, dummy_out,
                               NULL, 1, n / 2 + 1, FFTW_ESTIMATE);
    fftw_free(dummy_in);
    fftw_free(dummy_out);
  } else if (kind == FFT_KIND_C2R) {
    fftw_complex *dummy_in =
        (fftw_complex *)fftw_malloc((n / 2 + 1) * batch * sizeof(fftw_complex));
    double *dummy_out = (double *)fftw_malloc(n * batch * sizeof(double));
    cache->plan_f64 =
        fftw_plan_many_dft_c2r(1, &n, batch, dummy_in, NULL, 1, n / 2 + 1,
                               dummy_out, NULL, 1, n, FFTW_ESTIMATE);
    fftw_free(dummy_in);
    fftw_free(dummy_out);
  } else {
    fftw_complex *dummy_in =
        (fftw_complex *)fftw_malloc(n * batch * sizeof(fftw_complex));
    fftw_complex *dummy_out =
        (fftw_complex *)fftw_malloc(n * batch * sizeof(fftw_complex));
    cache->plan_f64 =
        fftw_plan_many_dft(1, &n, batch, dummy_in, NULL, 1, n, dummy_out, NULL,
                           1, n, direction, FFTW_ESTIMATE);
    fftw_free(dummy_in);
    fftw_free(dummy_out);
  }

  cache->n = n;
  cache->batch = batch;
  cache->direction = direction;
  cache->kind = kind;

  return cache->plan_f64;
}

fftwf_plan fft_ensure_plan_f32(int n, int64_t batch, int direction,
                               FFTKind kind) {
  FFTWCache *cache = fft_get_cache();

  if (cache->plan_f32 && cache->n == n && cache->batch == batch &&
      cache->direction == direction && cache->kind == kind) {
    return cache->plan_f32;
  }

  // Destroy old plan
  if (cache->plan_f32) {
    fftwf_destroy_plan(cache->plan_f32);
  }

  // Create new plan
  if (kind == FFT_KIND_R2C) {
    float *dummy_in = (float *)fftwf_malloc(n * batch * sizeof(float));
    fftwf_complex *dummy_out = (fftwf_complex *)fftwf_malloc(
        (n / 2 + 1) * batch * sizeof(fftwf_complex));
    cache->plan_f32 =
        fftwf_plan_many_dft_r2c(1, &n, batch, dummy_in, NULL, 1, n, dummy_out,
                                NULL, 1, n / 2 + 1, FFTW_ESTIMATE);
    fftwf_free(dummy_in);
    fftwf_free(dummy_out);
  } else if (kind == FFT_KIND_C2R) {
    fftwf_complex *dummy_in = (fftwf_complex *)fftwf_malloc(
        (n / 2 + 1) * batch * sizeof(fftwf_complex));
    float *dummy_out = (float *)fftwf_malloc(n * batch * sizeof(float));
    cache->plan_f32 =
        fftwf_plan_many_dft_c2r(1, &n, batch, dummy_in, NULL, 1, n / 2 + 1,
                                dummy_out, NULL, 1, n, FFTW_ESTIMATE);
    fftwf_free(dummy_in);
    fftwf_free(dummy_out);
  } else {
    fftwf_complex *dummy_in =
        (fftwf_complex *)fftwf_malloc(n * batch * sizeof(fftwf_complex));
    fftwf_complex *dummy_out =
        (fftwf_complex *)fftwf_malloc(n * batch * sizeof(fftwf_complex));
    cache->plan_f32 =
        fftwf_plan_many_dft(1, &n, batch, dummy_in, NULL, 1, n, dummy_out, NULL,
                            1, n, direction, FFTW_ESTIMATE);
    fftwf_free(dummy_in);
    fftwf_free(dummy_out);
  }

  cache->n = n;
  cache->batch = batch;
  cache->direction = direction;
  cache->kind = kind;

  return cache->plan_f32;
}
}

#endif