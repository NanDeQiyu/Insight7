// backends/cuda/kernels/fft/common.cuh
/**
 * @file common.cuh
 * @brief Common utilities for CUDA FFT kernels.
 *
 * Provides cuFFT plan caching and error handling for FFT operations.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string>

/**
 * @brief Type of FFT transform.
 */
enum class CuFFTKind {
  C2C = 0, ///< Complex to complex
  R2C = 1, ///< Real to complex
  C2R = 2  ///< Complex to real
};

/**
 * @brief Thread-local cuFFT plan cache.
 *
 * Caches the most recently used plan to avoid repeated creation overhead.
 * Each thread has its own cache instance.
 */
struct CuFFTPlanCache {
  cufftHandle plan = 0;
  int64_t n = 0;
  int64_t batch = 0;
  int direction = 0; // CUFFT_FORWARD or CUFFT_INVERSE
  CuFFTKind kind = CuFFTKind::C2C;
  bool is_f32 = true;

  /**
   * @brief Check if cache matches given parameters.
   */
  bool matches(int64_t n_, int64_t batch_, int direction_, CuFFTKind kind_,
               bool is_f32_) const {
    return n == n_ && batch == batch_ && direction == direction_ &&
           kind == kind_ && is_f32 == is_f32_;
  }

  /**
   * @brief Destroy the cached plan.
   */
  void destroy() {
    if (plan != 0) {
      cufftDestroy(plan);
      plan = 0;
    }
    n = 0;
    batch = 0;
    direction = 0;
    kind = CuFFTKind::C2C;
    is_f32 = true;
  }
};

/**
 * @brief Get thread-local cuFFT plan cache.
 *
 * @return Reference to thread-local cache
 */
inline CuFFTPlanCache &cufft_get_cache() {
  thread_local CuFFTPlanCache cache;
  return cache;
}

/**
 * @brief Ensure cuFFT plan exists for given parameters.
 *
 * Creates or reuses a cached cuFFT plan. The plan is invalidated if parameters
 * change.
 *
 * @param n        FFT length
 * @param batch    Number of transforms
 * @param direction CUFFT_FORWARD or CUFFT_INVERSE
 * @param kind     Transform kind (C2C, R2C, C2R)
 * @param is_f32   True for single precision, false for double precision
 * @return cufftHandle Valid plan handle
 */
inline cufftHandle cufft_ensure_plan(int64_t n, int64_t batch, int direction,
                                     CuFFTKind kind, bool is_f32) {
  CuFFTPlanCache &cache = cufft_get_cache();

  // Return cached plan if parameters match
  if (cache.plan != 0 && cache.matches(n, batch, direction, kind, is_f32)) {
    return cache.plan;
  }

  // Destroy old plan if exists
  cache.destroy();

  // Create new plan
  cufftHandle plan;
  cufftResult result;

  if (batch == 1) {
    // Single transform
    if (kind == CuFFTKind::C2C) {
      if (is_f32) {
        result = cufftPlan1d(&plan, static_cast<int>(n), CUFFT_C2C, 1);
      } else {
        result = cufftPlan1d(&plan, static_cast<int>(n), CUFFT_Z2Z, 1);
      }
    } else if (kind == CuFFTKind::R2C) {
      if (is_f32) {
        result = cufftPlan1d(&plan, static_cast<int>(n), CUFFT_R2C, 1);
      } else {
        result = cufftPlan1d(&plan, static_cast<int>(n), CUFFT_D2Z, 1);
      }
    } else { // C2R
      if (is_f32) {
        result = cufftPlan1d(&plan, static_cast<int>(n), CUFFT_C2R, 1);
      } else {
        result = cufftPlan1d(&plan, static_cast<int>(n), CUFFT_Z2D, 1);
      }
    }
  } else {
    // Batched transform
    int n_int = static_cast<int>(n);
    int batch_int = static_cast<int>(batch);

    if (kind == CuFFTKind::C2C) {
      if (is_f32) {
        result = cufftPlanMany(&plan, 1, &n_int, nullptr, 1,
                               n_int,             // input stride and dist
                               nullptr, 1, n_int, // output stride and dist
                               CUFFT_C2C, batch_int);
      } else {
        result = cufftPlanMany(&plan, 1, &n_int, nullptr, 1, n_int, nullptr, 1,
                               n_int, CUFFT_Z2Z, batch_int);
      }
    } else if (kind == CuFFTKind::R2C) {
      // Real input: n real values per transform
      // Complex output: n/2+1 complex values per transform
      int out_dist = n / 2 + 1;
      if (is_f32) {
        result = cufftPlanMany(&plan, 1, &n_int, nullptr, 1, n_int, // input
                               nullptr, 1, out_dist,                // output
                               CUFFT_R2C, batch_int);
      } else {
        result = cufftPlanMany(&plan, 1, &n_int, nullptr, 1, n_int, nullptr, 1,
                               out_dist, CUFFT_D2Z, batch_int);
      }
    } else { // C2R
      // Complex input: n/2+1 complex values per transform
      // Real output: n real values per transform
      int in_dist = n / 2 + 1;
      if (is_f32) {
        result = cufftPlanMany(&plan, 1, &n_int, nullptr, 1, in_dist, // input
                               nullptr, 1, n_int,                     // output
                               CUFFT_C2R, batch_int);
      } else {
        result = cufftPlanMany(&plan, 1, &n_int, nullptr, 1, in_dist, nullptr,
                               1, n_int, CUFFT_Z2D, batch_int);
      }
    }
  }

  if (result != CUFFT_SUCCESS) {
    return 0; // Error
  }

  // Update cache
  cache.plan = plan;
  cache.n = n;
  cache.batch = batch;
  cache.direction = direction;
  cache.kind = kind;
  cache.is_f32 = is_f32;

  return plan;
}

/**
 * @brief Get cuFFT error string.
 *
 * @param error cuFFT error code
 * @return Human-readable error string
 */
inline const char *cufft_get_error_string(cufftResult error) {
  switch (error) {
  case CUFFT_SUCCESS:
    return "cuFFT success";
  case CUFFT_INVALID_PLAN:
    return "cuFFT invalid plan";
  case CUFFT_ALLOC_FAILED:
    return "cuFFT allocation failed";
  case CUFFT_INVALID_TYPE:
    return "cuFFT invalid type";
  case CUFFT_INVALID_VALUE:
    return "cuFFT invalid value";
  case CUFFT_INTERNAL_ERROR:
    return "cuFFT internal error";
  case CUFFT_EXEC_FAILED:
    return "cuFFT execution failed";
  case CUFFT_SETUP_FAILED:
    return "cuFFT setup failed";
  case CUFFT_INVALID_SIZE:
    return "cuFFT invalid size";
  case CUFFT_UNALIGNED_DATA:
    return "cuFFT unaligned data";
  default:
    return "cuFFT unknown error";
  }
}

/**
 * @brief Set GPU error message from cuFFT result.
 *
 * @param op     Operation name
 * @param result cuFFT error code
 */
inline void cufft_set_error(const char *op, cufftResult result) {
  thread_local char err_buf[256];
  snprintf(err_buf, sizeof(err_buf), "%s: %s", op,
           cufft_get_error_string(result));
  gpu_set_last_error(err_buf);
}

/**
 * @brief Set GPU error message from string.
 *
 * @param msg Error message
 */
inline void cufft_set_error(const char *msg) { gpu_set_last_error(msg); }
