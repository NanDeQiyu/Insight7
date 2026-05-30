// backends/cpu/kernels/fft/common.c
/**
 * @file common.c
 * @brief FFTW plan cache implementation.
 *
 * Thread-local plan caching for FFTW. For C2C transforms, plans are created
 * with the actual data buffer (required for correctness at n>=64). For R2C/C2R,
 * dummy buffers are used (these transforms work correctly with
 * fftw_execute_dft).
 */

#ifdef INSIGHT_USE_FFTW3

#include "common.h"
#include <string.h>

// Thread-local FFTW plan cache
static thread_local FFTWCache cache = {NULL, NULL, NULL, NULL,
                                       0,    0,    0,    FFT_KIND_C2C};

FFTWCache *fft_get_cache(void) { return &cache; }

fftw_plan fft_ensure_plan_f64(int n, int64_t batch, int direction, FFTKind kind,
                              void *buf) {
  FFTWCache *c = fft_get_cache();

  // Check if cached plan matches AND buffer matches (for C2C)
  int need_recreate = (c->n != n || c->batch != batch ||
                       c->direction != direction || c->kind != kind);
  if (!need_recreate && kind == FFT_KIND_C2C) {
    need_recreate = (c->buf_f64 != buf);
  }

  if (!need_recreate && c->plan_f64) {
    return c->plan_f64;
  }

  // Destroy old plan
  if (c->plan_f64) {
    fftw_destroy_plan(c->plan_f64);
    c->plan_f64 = NULL;
  }

  int n_int = (int)n;

  if (kind == FFT_KIND_C2C) {
    // C2C: create plan with actual buffer (required for correctness)
    c->plan_f64 = fftw_plan_many_dft(1, &n_int, (int)batch, (fftw_complex *)buf,
                                     NULL, 1, n_int, (fftw_complex *)buf, NULL,
                                     1, n_int, direction, FFTW_ESTIMATE);
  } else {
    // R2C/C2R: create with dummy buffers (these transforms work correctly
    // with fftw_execute_dft_r2c/c2r on different buffers)
    fftw_complex *dummy_in =
        (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));
    fftw_complex *dummy_out =
        (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));

    if (kind == FFT_KIND_R2C) {
      c->plan_f64 = fftw_plan_many_dft_r2c(
          1, &n_int, (int)batch, (double *)dummy_in, NULL, 1, n_int, dummy_out,
          NULL, 1, n_int / 2 + 1, FFTW_ESTIMATE);
    } else {
      c->plan_f64 = fftw_plan_many_dft_c2r(
          1, &n_int, (int)batch, dummy_in, NULL, 1, n_int / 2 + 1,
          (double *)dummy_out, NULL, 1, n_int, FFTW_ESTIMATE);
    }

    fftw_free(dummy_in);
    fftw_free(dummy_out);
  }

  c->buf_f64 = buf;
  c->n = n;
  c->batch = batch;
  c->direction = direction;
  c->kind = kind;
  return c->plan_f64;
}

fftwf_plan fft_ensure_plan_f32(int n, int64_t batch, int direction,
                               FFTKind kind, void *buf) {
  FFTWCache *c = fft_get_cache();

  int need_recreate = (c->n != n || c->batch != batch ||
                       c->direction != direction || c->kind != kind);
  if (!need_recreate) {
    need_recreate = (c->buf_f32 != buf);
  }

  if (!need_recreate && c->plan_f32) {
    return c->plan_f32;
  }

  if (c->plan_f32) {
    fftwf_destroy_plan(c->plan_f32);
    c->plan_f32 = NULL;
  }

  int n_int = (int)n;

  if (kind == FFT_KIND_C2C) {
    c->plan_f32 = fftwf_plan_many_dft(
        1, &n_int, (int)batch, (fftwf_complex *)buf, NULL, 1, n_int,
        (fftwf_complex *)buf, NULL, 1, n_int, direction, FFTW_ESTIMATE);
  } else {
    // R2C/C2R: use dummy buffers
    fftwf_complex *dummy_in =
        (fftwf_complex *)fftwf_malloc(n * sizeof(fftwf_complex));
    fftwf_complex *dummy_out =
        (fftwf_complex *)fftwf_malloc(n * sizeof(fftwf_complex));

    if (kind == FFT_KIND_R2C) {
      c->plan_f32 = fftwf_plan_many_dft_r2c(
          1, &n_int, (int)batch, (float *)dummy_in, NULL, 1, n_int, dummy_out,
          NULL, 1, n_int / 2 + 1, FFTW_ESTIMATE);
    } else {
      c->plan_f32 = fftwf_plan_many_dft_c2r(
          1, &n_int, (int)batch, dummy_in, NULL, 1, n_int / 2 + 1,
          (float *)dummy_out, NULL, 1, n_int, FFTW_ESTIMATE);
    }

    fftwf_free(dummy_in);
    fftwf_free(dummy_out);
  }

  c->buf_f32 = buf;
  c->n = n;
  c->batch = batch;
  c->direction = direction;
  c->kind = kind;
  return c->plan_f32;
}

#endif // INSIGHT_USE_FFTW3
