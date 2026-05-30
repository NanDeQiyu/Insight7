// src/ops/signal/convolution.cpp
#include "insight/ops/signal/convolution.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/reduction.h"
#include "insight/ops/unary.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

namespace ins {
namespace signal {

namespace {

// Helper: extract center portion of array (for "same" and "valid" modes)
Array center_crop(const Array &arr, const Shape &target_shape) {
  // For each dimension, compute start offset to center-crop
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  for (int i = 0; i < arr.shape().ndim(); ++i) {
    int64_t diff = arr.shape().dim(i) - target_shape.dim(i);
    int64_t start = diff / 2;
    starts.push_back(start);
    ends.push_back(start + target_shape.dim(i));
  }

  // Use slice for each dimension
  Array result = arr;
  for (int i = 0; i < arr.shape().ndim(); ++i) {
    result = slice(result, {i}, {starts[i]}, {ends[i]});
  }
  return result;
}

// Compute output shape for convolution
Shape conv_output_shape(const Shape &s1, const Shape &s2,
                        const std::string &mode) {
  int ndim = std::max(s1.ndim(), s2.ndim());
  std::vector<int64_t> out_shape(ndim);

  for (int i = 0; i < ndim; ++i) {
    int64_t d1 = (i < s1.ndim()) ? s1.dim(i) : 1;
    int64_t d2 = (i < s2.ndim()) ? s2.dim(i) : 1;
    int64_t full_size = d1 + d2 - 1;

    if (mode == "full") {
      out_shape[i] = full_size;
    } else if (mode == "same") {
      out_shape[i] = std::max(d1, d2);
    } else { // "valid"
      out_shape[i] = std::max(d1, d2) - std::min(d1, d2) + 1;
    }
  }
  return Shape(out_shape);
}

// Reverse array along all dimensions
Array reverse_all(const Array &x) {
  Array result = x;
  for (int i = 0; i < x.shape().ndim(); ++i) {
    result = flip(result, i);
  }
  return result;
}

} // anonymous namespace

// ============================================================================
// fftconvolve
// ============================================================================

Array fftconvolve(const Array &in1, const Array &in2, const std::string &mode) {
  INS_CHECK(in1.defined() && in2.defined(),
            "fftconvolve: inputs are undefined");
  INS_CHECK(mode == "full" || mode == "same" || mode == "valid",
            "fftconvolve: mode must be 'full', 'same', or 'valid'");

  int ndim = std::max(in1.shape().ndim(), in2.shape().ndim());

  // Determine working dtype
  DType work_dtype = DType::F64;
  bool is_complex = (in1.dtype() == DType::C32 || in1.dtype() == DType::C64 ||
                     in2.dtype() == DType::C32 || in2.dtype() == DType::C64);
  if (is_complex) {
    work_dtype = DType::C64;
    if (in1.dtype() == DType::C32 && in2.dtype() == DType::C32)
      work_dtype = DType::C32;
  } else if (in1.dtype() == DType::F32 && in2.dtype() == DType::F32) {
    work_dtype = DType::F32;
  }

  Array a = (in1.dtype() == work_dtype) ? in1 : in1.to(work_dtype);
  Array b = (in2.dtype() == work_dtype) ? in2 : in2.to(work_dtype);

  if (ndim == 1) {
    // 1D FFT convolution
    int64_t n = a.numel();
    int64_t m = b.numel();
    int64_t conv_len = n + m - 1;
    int64_t fft_len = fft::next_fast_len(conv_len);

    Array result;
    if (is_complex) {
      Array A = fft::fft(a, fft_len);
      Array B = fft::fft(b, fft_len);
      Array C = mul(A, B);
      result = fft::ifft(C, conv_len);
    } else {
      Array A = fft::rfft(a, fft_len);
      Array B = fft::rfft(b, fft_len);
      Array C = mul(A, B);
      result = fft::irfft(C, conv_len);
    }

    // Crop for mode
    Shape target = conv_output_shape(a.shape(), b.shape(), mode);
    if (mode != "full") {
      result = center_crop(result, target);
    }

    if (result.dtype() != in1.dtype()) {
      result = result.to(in1.dtype());
    }
    return result;
  }

  // N-D FFT convolution
  // Compute full output shape
  std::vector<int64_t> full_shape;
  std::vector<int64_t> fft_shape;
  for (int i = 0; i < ndim; ++i) {
    int64_t d1 = (i < a.shape().ndim()) ? a.shape().dim(i) : 1;
    int64_t d2 = (i < b.shape().ndim()) ? b.shape().dim(i) : 1;
    int64_t full_size = d1 + d2 - 1;
    full_shape.push_back(full_size);
    fft_shape.push_back(fft::next_fast_len(full_size));
  }

  // Use rfftn for real inputs, fftn for complex
  Array result;
  if (is_complex) {
    Array A = fft::fftn(a, fft_shape);
    Array B = fft::fftn(b, fft_shape);
    Array C = mul(A, B);
    result = fft::ifftn(C, full_shape);
  } else {
    Array A = fft::rfftn(a, fft_shape);
    Array B = fft::rfftn(b, fft_shape);
    Array C = mul(A, B);
    result = fft::irfftn(C, full_shape);
  }

  // Crop for mode
  Shape target = conv_output_shape(a.shape(), b.shape(), mode);
  if (mode != "full") {
    result = center_crop(result, target);
  }

  if (result.dtype() != in1.dtype()) {
    result = result.to(in1.dtype());
  }
  return result;
}

// ============================================================================
// correlate
// ============================================================================

Array correlate(const Array &in1, const Array &in2, const std::string &mode) {
  INS_CHECK(in1.defined() && in2.defined(), "correlate: inputs are undefined");
  INS_CHECK(mode == "full" || mode == "same" || mode == "valid",
            "correlate: mode must be 'full', 'same', or 'valid'");

  // Cross-correlation = convolution with reversed (and conjugated) in2
  Array in2_rev = reverse_all(in2);

  // For complex inputs, take conjugate
  if (in2.dtype() == DType::C32 || in2.dtype() == DType::C64) {
    in2_rev = conj(in2_rev);
  }

  return fftconvolve(in1, in2_rev, mode);
}

// ============================================================================
// convolve2d
// ============================================================================

Array convolve2d(const Array &in1, const Array &in2, const std::string &mode) {
  INS_CHECK(in1.defined() && in2.defined(), "convolve2d: inputs are undefined");
  INS_CHECK(in1.shape().ndim() == 2 && in2.shape().ndim() == 2,
            "convolve2d: inputs must be 2D");
  INS_CHECK(mode == "full" || mode == "same" || mode == "valid",
            "convolve2d: mode must be 'full', 'same', or 'valid'");

  return fftconvolve(in1, in2, mode);
}

// ============================================================================
// correlate2d
// ============================================================================

Array correlate2d(const Array &in1, const Array &in2, const std::string &mode) {
  INS_CHECK(in1.defined() && in2.defined(),
            "correlate2d: inputs are undefined");
  INS_CHECK(in1.shape().ndim() == 2 && in2.shape().ndim() == 2,
            "correlate2d: inputs must be 2D");
  INS_CHECK(mode == "full" || mode == "same" || mode == "valid",
            "correlate2d: mode must be 'full', 'same', or 'valid'");

  // Reverse in2 along both dimensions
  Array in2_rev = flip(flip(in2, 0), 1);

  // For complex inputs, take conjugate
  if (in2.dtype() == DType::C32 || in2.dtype() == DType::C64) {
    in2_rev = conj(in2_rev);
  }

  return fftconvolve(in1, in2_rev, mode);
}

// ============================================================================
// choose_conv_method
// ============================================================================

std::string choose_conv_method(const Array &in1, const Array &in2,
                               const std::string &mode) {
  INS_CHECK(in1.defined() && in2.defined(),
            "choose_conv_method: inputs are undefined");

  // Heuristic: for small arrays, direct is faster; for large, FFT
  // Threshold based on total output size
  int64_t out_size = 1;
  for (int i = 0; i < in1.shape().ndim(); ++i) {
    int64_t d1 = in1.shape().dim(i);
    int64_t d2 = (i < in2.shape().ndim()) ? in2.shape().dim(i) : 1;
    if (mode == "full") {
      out_size *= (d1 + d2 - 1);
    } else if (mode == "same") {
      out_size *= std::max(d1, d2);
    } else {
      out_size *= std::max(d1, d2) - std::min(d1, d2) + 1;
    }
  }

  // FFT is generally faster for larger sizes
  // Direct convolution is O(n*m), FFT is O(N*log(N))
  // Crossover point: roughly when output_size > 256
  if (out_size > 256) {
    return "fft";
  }
  return "direct";
}

// ============================================================================
// correlation_lags
// ============================================================================

Array correlation_lags(int64_t in1_len, int64_t in2_len,
                       const std::string &mode) {
  INS_CHECK(in1_len >= 1 && in2_len >= 1,
            "correlation_lags: lengths must be >= 1");
  INS_CHECK(mode == "full" || mode == "same" || mode == "valid",
            "correlation_lags: mode must be 'full', 'same', or 'valid'");

  int64_t lag_start, lag_end;

  if (mode == "full") {
    // Lags from -(in2_len-1) to (in1_len-1)
    lag_start = -(in2_len - 1);
    lag_end = in1_len - 1;
  } else if (mode == "same") {
    // Center portion of full lags
    int64_t full_len = in1_len + in2_len - 1;
    int64_t out_len = std::max(in1_len, in2_len);
    int64_t start = (full_len - out_len) / 2;
    lag_start = -(in2_len - 1) + start;
    lag_end = lag_start + out_len - 1;
  } else { // "valid"
    // Only where arrays fully overlap
    if (in1_len >= in2_len) {
      lag_start = -(in2_len - 1) + (in2_len - 1);
      lag_end = in1_len - in2_len;
    } else {
      lag_start = -(in2_len - 1) + (in2_len - 1);
      lag_end = in1_len - 1;
    }
  }

  int64_t n = lag_end - lag_start + 1;
  return arange(static_cast<double>(lag_start),
                static_cast<double>(lag_end + 1), 1.0, DType::I64);
}

} // namespace signal
} // namespace ins
