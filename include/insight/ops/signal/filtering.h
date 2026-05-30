// insight/ops/signal/filtering.h
#pragma once
#include "insight/core/array.h"
#include <string>
#include <vector>

namespace ins {
namespace signal {

/**
 * @brief Compute the analytic signal using the Hilbert transform.
 *
 * The analytic signal is x + j*Hilbert(x), where H(x) is the Hilbert
 * transform. The imaginary part is the Hilbert transform of the real input.
 *
 * @param x Real input array (1D)
 * @param N FFT size. Default: x.numel()
 * @return Complex analytic signal (C64 or C32)
 */
Array hilbert(const Array &x, int64_t N = -1);

/**
 * @brief Compute the 2-D analytic signal using the Hilbert transform.
 *
 * @param x Real 2D input array
 * @param N FFT size for each dimension. Default: use input dimensions
 * @return Complex 2-D analytic signal
 */
Array hilbert2(const Array &x, int64_t N = -1);

/**
 * @brief Remove linear or constant trend from data.
 *
 * @param data Input array
 * @param axis Axis along which to detrend. Default: -1
 * @param type "linear" or "constant". Default: "linear"
 * @return Detrended array (same shape as input)
 */
Array detrend(const Array &data, int axis = -1,
              const std::string &type = "linear");

/**
 * @brief Wiener filter for N-D arrays.
 *
 * Applies a Wiener filter to smooth/denoise the input. Estimates local
 * mean and variance using a sliding window.
 *
 * @param im Input array
 * @param mysize Size of the sliding window for each dimension.
 *               Default: 3 for each dimension
 * @param noise Noise power. Default: estimated from data
 * @return Filtered array (same shape as im)
 */
Array wiener(const Array &im, const std::vector<int64_t> &mysize = {},
             double noise = -1.0);

/**
 * @brief FIR filter (direct convolution).
 *
 * Applies an FIR filter b to signal x along the specified axis.
 *
 * @param b FIR filter coefficients (1D)
 * @param x Input signal
 * @param axis Axis along which to filter. Default: -1
 * @return Filtered signal (same shape as x)
 */
Array firfilter(const Array &b, const Array &x, int axis = -1);

/**
 * @brief FIR filter with initial conditions.
 *
 * @param b FIR filter coefficients (1D)
 * @param x Input signal
 * @param zi Initial filter state (length len(b)-1). If empty, zeros used.
 * @param axis Axis along which to filter. Default: -1
 * @return Pair of {filtered signal, final filter state}
 */
std::pair<Array, Array> firfilter_zi_state(const Array &b, const Array &x,
                                           const Array &zi, int axis = -1);

/**
 * @brief IIR/FIR digital filter.
 *
 * For FIR (a has length 1), delegates to firfilter.
 * For IIR (a has length > 1), uses direct-form II transpose.
 *
 * @param b Numerator coefficients (1D)
 * @param a Denominator coefficients (1D). a[0] must be 1.
 * @param x Input signal
 * @param axis Axis along which to filter. Default: -1
 * @return Filtered signal
 */
Array lfilter(const Array &b, const Array &a, const Array &x, int axis = -1);

/**
 * @brief Compute initial conditions for lfilter for step-response steady state.
 *
 * @param b Numerator coefficients (1D)
 * @param a Denominator coefficients (1D)
 * @return Initial condition array (length max(len(b), len(a)) - 1)
 */
Array lfilter_zi(const Array &b, const Array &a);

/**
 * @brief Zero-phase digital filter (forward-backward filtering).
 *
 * Applies the filter twice: once forward and once backward, to achieve
 * zero phase distortion.
 *
 * @param b Numerator coefficients (1D)
 * @param a Denominator coefficients (1D)
 * @param x Input signal
 * @param axis Axis along which to filter. Default: -1
 * @return Filtered signal (same shape as x)
 */
Array filtfilt(const Array &b, const Array &a, const Array &x, int axis = -1);

/**
 * @brief Downsample with anti-aliasing filter.
 *
 * @param x Input signal
 * @param q Downsampling factor (integer)
 * @param axis Axis along which to decimate. Default: -1
 * @param zero_phase If true, use forward-backward filtering (filtfilt).
 *                   Default: true
 * @return Downsampled signal
 */
Array decimate(const Array &x, int64_t q, int axis = -1,
               bool zero_phase = true);

/**
 * @brief Resample signal using FFT-based method.
 *
 * @param x Input signal
 * @param num Number of output samples
 * @param axis Axis along which to resample. Default: -1
 * @return Resampled signal
 */
Array resample(const Array &x, int64_t num, int axis = -1);

/**
 * @brief Resample signal using polyphase filtering.
 *
 * @param x Input signal
 * @param upsampling factor
 * @param downsampling factor
 * @param axis Axis along which to resample. Default: -1
 * @return Resampled signal
 */
Array resample_poly(const Array &x, int64_t up, int64_t down, int axis = -1);

/**
 * @brief Frequency shift signal by multiplying with complex exponential.
 *
 * @param x Input signal (real)
 * @param freq Frequency shift in Hz
 * @param fs Sampling frequency in Hz
 * @return Frequency-shifted signal (complex)
 */
Array freq_shift(const Array &x, double freq, double fs);

} // namespace signal
} // namespace ins
