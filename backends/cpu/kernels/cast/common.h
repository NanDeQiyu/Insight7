// backends/cpu/kernels/cast/common.h
/**
 * @file common.h
 * @brief Common utilities for CPU cast kernels.
 *
 * Provides layout copying and helper functions for casting between types.
 */

#pragma once
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <complex>
#include <cstring>

// Copy layout information from source to destination
static inline void copy_layout(InsightArray *dst, const InsightArray *src) {
  dst->ndim = src->ndim;
  dst->numel = src->numel;
  dst->offset = src->offset;
  for (int i = 0; i < src->ndim; ++i) {
    dst->dims[i] = src->dims[i];
    dst->strides[i] = src->strides[i];
  }
}

// Helper: copy data with OpenMP parallel for
#define CAST_LOOP(n, code)                                                     \
  do {                                                                         \
    _Pragma("omp parallel for") for (int64_t i = 0; i < n; ++i) { code; }      \
  } while (0)
