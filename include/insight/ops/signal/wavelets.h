// insight/ops/signal/wavelets.h
#pragma once
#include "insight/core/array.h"
#include <functional>

namespace ins {
namespace signal {

/// @brief Complex Morlet wavelet.
///
/// @param M Length of the wavelet
/// @param w Omega0. Default: 5.0
/// @param s Scaling factor. Default: 1.0
/// @param complete Use complete (admissibility-corrected) version. Default:
/// true
/// @return Complex128 array of length M
Array morlet(int64_t M, double w = 5.0, double s = 1.0, bool complete = true);

/// @brief Ricker wavelet (Mexican hat wavelet).
///
/// @param points Number of points in the output
/// @param a Width parameter
/// @return Float64 array of length points
Array ricker(int64_t points, double a);

/// @brief Complex Morlet wavelet compatible with cwt.
///
/// @param M Length of the wavelet
/// @param s Width parameter
/// @param w Omega0. Default: 5.0
/// @return Complex128 array of length M
Array morlet2(int64_t M, double s, double w = 5.0);

/// @brief Continuous Wavelet Transform.
///
/// @param data 1D input signal
/// @param wavelet Wavelet function taking (length, width) and returning Array
/// @param widths Array of widths to use for transform
/// @return 2D array of shape (len(widths), len(data))
Array cwt(const Array &data, std::function<Array(int64_t, double)> wavelet,
          const std::vector<double> &widths);

} // namespace signal
} // namespace ins
