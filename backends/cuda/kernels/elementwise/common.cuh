// backends/cuda/kernels/elementwise/common.cuh
/**
 * @file common.cuh
 * @brief Common utilities for CUDA elementwise kernels.
 */

#pragma once
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

static inline dim3 elementwise_blocks(int64_t n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  return dim3(blocks);
}

static inline dim3 elementwise_threads() { return dim3(256); }

struct ElementwiseMetadata {
  int64_t ndim;
  int64_t numel;
  int64_t dims[INSIGHT_MAX_NDIM];
  int64_t a_strides[INSIGHT_MAX_NDIM];
  int64_t b_strides[INSIGHT_MAX_NDIM];
  int64_t out_strides[INSIGHT_MAX_NDIM];
};

static inline ElementwiseMetadata *
alloc_elementwise_metadata(const InsightArray *a, const InsightArray *b,
                           const InsightArray *out) {

  ElementwiseMetadata *d_meta;
  cudaMalloc(&d_meta, sizeof(ElementwiseMetadata));

  ElementwiseMetadata h_meta;
  h_meta.ndim = out->ndim;
  h_meta.numel = out->numel;

  int b_ndim = b->ndim;
  int b_offset = h_meta.ndim - b_ndim;

  for (int i = 0; i < h_meta.ndim; ++i) {
    h_meta.dims[i] = out->dims[i];
    h_meta.a_strides[i] = a->strides[i];
    h_meta.out_strides[i] = out->strides[i];

    int b_i = i - b_offset;
    if (b_i >= 0 && b_i < b_ndim && b->dims[b_i] == out->dims[i]) {
      h_meta.b_strides[i] = b->strides[b_i];
    } else {
      h_meta.b_strides[i] = 0;
    }
  }

  cudaMemcpy(d_meta, &h_meta, sizeof(ElementwiseMetadata),
             cudaMemcpyHostToDevice);
  return d_meta;
}

static inline void free_elementwise_metadata(ElementwiseMetadata *meta) {
  if (meta)
    cudaFree(meta);
}

__device__ static inline int64_t
elementwise_offset(int64_t linear, const ElementwiseMetadata *meta,
                   const int64_t *strides) {

  int64_t offset = 0;
  int64_t tmp = linear;
  for (int d = meta->ndim - 1; d >= 0; --d) {
    int64_t idx = tmp % meta->dims[d];
    tmp /= meta->dims[d];
    offset += idx * strides[d];
  }
  return offset;
}
