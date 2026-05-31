// backends/cpu/kernels/fft/common_impl.cpp
/**
 * @file common_impl.cpp
 * @brief FFTW plan cache implementation.
 *
 * - C2C: plans created with actual buffer each time (FFTW_ESTIMATE +
 * fftw_execute). fftw_execute_dft gives wrong results for n>=64 with
 * FFTW_ESTIMATE.
 * - R2C/C2R: plans cached with dummy buffers (fftw_execute_dft_r2c/c2r works).
 */

#ifdef INSIGHT_USE_FFTW3

#include "common.h"
#include <cstring>

static thread_local FFTWCache cache = {NULL, NULL, NULL, NULL,
                                       0,    0,    0,    FFT_KIND_C2C};

FFTWCache *fft_get_cache(void) { return &cache; }

fftw_plan fft_ensure_plan_f64(int n, int64_t batch, int direction, FFTKind kind,
                              void *buf) {
  FFTWCache *c = &cache;

  if (kind == FFT_KIND_C2C) {
    if (c->plan_f64 && c->kind == FFT_KIND_C2C) {
      fftw_destroy_plan(c->plan_f64);
      c->plan_f64 = NULL;
    }
    int n_int = (int)n;
    c->plan_f64 = fftw_plan_many_dft(1, &n_int, (int)batch, (fftw_complex *)buf,
                                     NULL, 1, n_int, (fftw_complex *)buf, NULL,
                                     1, n_int, direction, FFTW_ESTIMATE);
    c->n = n;
    c->batch = batch;
    c->direction = direction;
    c->kind = FFT_KIND_C2C;
    return c->plan_f64;
  }

  // R2C/C2R: cache with dummy buffers
  if (c->plan_f64 && c->n == n && c->batch == batch &&
      c->direction == direction && c->kind == kind) {
    return c->plan_f64;
  }

  if (c->plan_f64) {
    fftw_destroy_plan(c->plan_f64);
    c->plan_f64 = NULL;
  }

  int n_int = (int)n;
  fftw_complex *di = (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));
  fftw_complex *dout = (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));

  if (kind == FFT_KIND_R2C) {
    c->plan_f64 = fftw_plan_many_dft_r2c(1, &n_int, (int)batch, (double *)di,
                                         NULL, 1, n_int, dout, NULL, 1,
                                         n_int / 2 + 1, FFTW_ESTIMATE);
  } else {
    c->plan_f64 = fftw_plan_many_dft_c2r(1, &n_int, (int)batch, di, NULL, 1,
                                         n_int / 2 + 1, (double *)dout, NULL, 1,
                                         n_int, FFTW_ESTIMATE);
  }

  fftw_free(di);
  fftw_free(dout);

  c->n = n;
  c->batch = batch;
  c->direction = direction;
  c->kind = kind;
  return c->plan_f64;
}

fftwf_plan fft_ensure_plan_f32(int n, int64_t batch, int direction,
                               FFTKind kind, void *buf) {
  FFTWCache *c = &cache;

  if (kind == FFT_KIND_C2C) {
    if (c->plan_f32 && c->kind == FFT_KIND_C2C) {
      fftwf_destroy_plan(c->plan_f32);
      c->plan_f32 = NULL;
    }
    int n_int = (int)n;
    c->plan_f32 = fftwf_plan_many_dft(
        1, &n_int, (int)batch, (fftwf_complex *)buf, NULL, 1, n_int,
        (fftwf_complex *)buf, NULL, 1, n_int, direction, FFTW_ESTIMATE);
    c->n = n;
    c->batch = batch;
    c->direction = direction;
    c->kind = FFT_KIND_C2C;
    return c->plan_f32;
  }

  if (c->plan_f32 && c->n == n && c->batch == batch &&
      c->direction == direction && c->kind == kind) {
    return c->plan_f32;
  }

  if (c->plan_f32) {
    fftwf_destroy_plan(c->plan_f32);
    c->plan_f32 = NULL;
  }

  int n_int = (int)n;
  fftwf_complex *di = (fftwf_complex *)fftwf_malloc(n * sizeof(fftwf_complex));
  fftwf_complex *dout =
      (fftwf_complex *)fftwf_malloc(n * sizeof(fftwf_complex));

  if (kind == FFT_KIND_R2C) {
    c->plan_f32 = fftwf_plan_many_dft_r2c(1, &n_int, (int)batch, (float *)di,
                                          NULL, 1, n_int, dout, NULL, 1,
                                          n_int / 2 + 1, FFTW_ESTIMATE);
  } else {
    c->plan_f32 = fftwf_plan_many_dft_c2r(1, &n_int, (int)batch, di, NULL, 1,
                                          n_int / 2 + 1, (float *)dout, NULL, 1,
                                          n_int, FFTW_ESTIMATE);
  }

  fftwf_free(di);
  fftwf_free(dout);

  c->n = n;
  c->batch = batch;
  c->direction = direction;
  c->kind = kind;
  return c->plan_f32;
}

#endif // INSIGHT_USE_FFTW3
