// insight/ops/signal/filter_design.h
#pragma once
#include "insight/core/array.h"
#include <string>
#include <vector>

namespace ins {
namespace signal {

/**
 * @brief Compute the Kaiser window beta parameter from desired attenuation.
 *
 * Given a desired stopband attenuation (in dB), returns the Kaiser beta
 * shape parameter.
 *
 * @param a Desired attenuation in dB (positive value)
 * @return Kaiser beta parameter
 */
double kaiser_beta(double a);

/**
 * @brief Compute the approximate attenuation of a Kaiser FIR filter.
 *
 * Estimates the stopband attenuation from the number of taps and transition
 * bandwidth.
 *
 * @param numtaps Number of filter taps
 * @param width Normalized transition bandwidth (in radians/sample)
 * @return Approximate attenuation in dB
 */
double kaiser_atten(int64_t numtaps, double width);

/**
 * @brief FIR filter design using the window method.
 *
 * Designs a Type I or Type II linear-phase FIR filter using a windowed-sinc
 * approach.
 *
 * @param numtaps Number of filter taps (must be >= 1)
 * @param cutoff Cutoff frequency or vector of cutoff frequencies (normalized
 * 0..1)
 * @param window Window name or window array. Default: "hamming"
 * @param pass_zero If true, the first band is stop (lowpass/highpass).
 *                  Can also be "lowpass", "highpass", "bandpass", "bandstop".
 * @param scale If true, scale the filter to have unity gain at DC (lowpass)
 *              or Nyquist (highpass). Default: true
 * @return 1D Array of filter coefficients (length numtaps)
 */
Array firwin(int64_t numtaps, const std::vector<double> &cutoff,
             const std::string &window = "hamming",
             const std::string &pass_zero = "lowpass", bool scale = true);

/**
 * @brief FIR filter design from frequency-magnitude pairs.
 *
 * Designs an FIR filter by interpolating the desired frequency response
 * from (freq, gain) pairs and computing the impulse response via IFFT.
 *
 * @param numtaps Number of filter taps (must be >= 1)
 * @param freq Frequency breakpoints (normalized 0..1, must start at 0 and end
 * at 1)
 * @param gain Gain at each frequency breakpoint (same length as freq)
 * @param nfreqs Number of FFT points for interpolation. Default: next power of
 * 2 >= numtaps
 * @param window Window name. Default: "hamming"
 * @param antisymmetric If true, design a Type III or IV filter. Default: false
 * @return 1D Array of filter coefficients (length numtaps)
 */
Array firwin2(int64_t numtaps, const std::vector<double> &freq,
              const std::vector<double> &gain, int64_t nfreqs = 0,
              const std::string &window = "hamming",
              bool antisymmetric = false);

/**
 * @brief Sort complex roots by magnitude.
 *
 * @param p Array of complex values (C64 dtype)
 * @return Array of sorted complex values
 */
Array cmplx_sort(const Array &p);

} // namespace signal
} // namespace ins
