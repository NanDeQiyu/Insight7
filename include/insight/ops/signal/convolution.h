// insight/ops/signal/convolution.h
#pragma once
#include "insight/core/array.h"
#include <string>

namespace ins {
namespace signal {

/**
 * @brief N-D convolution using FFT.
 *
 * Computes the full convolution of two arrays using the FFT.
 * Supports "full", "same", and "valid" output modes.
 *
 * @param in1 First input array
 * @param in2 Second input array
 * @param mode Output mode: "full" (default), "same", or "valid"
 * @return Convolved array
 */
Array fftconvolve(const Array &in1, const Array &in2,
                  const std::string &mode = "full");

/**
 * @brief N-D cross-correlation.
 *
 * Computes the cross-correlation of two arrays. For FFT-based method,
 * this is equivalent to convolving in1 with the reversed and conjugated in2.
 *
 * @param in1 First input array
 * @param in2 Second input array
 * @param mode Output mode: "full" (default), "same", or "valid"
 * @return Cross-correlation array
 */
Array correlate(const Array &in1, const Array &in2,
                const std::string &mode = "full");

/**
 * @brief 2D convolution.
 *
 * @param in1 First 2D input array
 * @param in2 Second 2D input array
 * @param mode Output mode: "full" (default), "same", or "valid"
 * @return 2D convolved array
 */
Array convolve2d(const Array &in1, const Array &in2,
                 const std::string &mode = "full");

/**
 * @brief 2D cross-correlation.
 *
 * @param in1 First 2D input array
 * @param in2 Second 2D input array
 * @param mode Output mode: "full" (default), "same", or "valid"
 * @return 2D cross-correlation array
 */
Array correlate2d(const Array &in1, const Array &in2,
                  const std::string &mode = "full");

/**
 * @brief Choose the best convolution method (FFT or direct).
 *
 * Heuristically determines whether FFT-based or direct convolution
 * is faster for the given input sizes.
 *
 * @param in1 First input array
 * @param in2 Second input array
 * @param mode Convolution mode
 * @return "fft" or "direct"
 */
std::string choose_conv_method(const Array &in1, const Array &in2,
                               const std::string &mode = "full");

/**
 * @brief Compute lag indices for 1D cross-correlation.
 *
 * Returns an array of lag values corresponding to the output of
 * correlate(a, b, mode).
 *
 * @param in1_len Length of first input
 * @param in2_len Length of second input
 * @param mode Correlation mode: "full", "same", or "valid"
 * @return 1D Array of integer lag values
 */
Array correlation_lags(int64_t in1_len, int64_t in2_len,
                       const std::string &mode = "full");

} // namespace signal
} // namespace ins
