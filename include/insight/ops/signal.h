// insight/ops/signal.h
#pragma once
#include "insight/core/array.h"
#define _USE_MATH_DEFINES
#include <cmath>

// Sub-modules
#include "insight/ops/signal/acoustics.h"
#include "insight/ops/signal/bsplines.h"
#include "insight/ops/signal/convolution.h"
#include "insight/ops/signal/filter_design.h"
#include "insight/ops/signal/filtering.h"
#include "insight/ops/signal/spectral_analysis.h"
#include "insight/ops/signal/waveforms.h"
#include "insight/ops/signal/wavelets.h"
#include "insight/ops/signal/windows.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ins {

/**
 * @brief Unwrap radian phase by adding/subtracting multiples of 2π.
 *
 * Unwraps the phase array by detecting large jumps (default threshold = π/2)
 * and adding multiples of 2π to correct them.
 *
 * @param p Input array (phase in radians)
 * @param axis Axis along which to unwrap (default: -1)
 * @param discont Discontinuity threshold (default: π)
 * @param period Period of the function (default: 2π)
 * @return Unwrapped array, same shape as input
 *
 * Example:
 * @code
 * Array phase = arange(0, 4 * M_PI, M_PI);
 * Array unwrapped = unwrap(phase);
 * @endcode
 */
Array unwrap(const Array &p, int axis = -1, double discont = M_PI,
             double period = 2 * M_PI);

/**
 * @brief Normalized sinc function: sinc(x) = sin(πx) / (πx)
 *
 * sinc(0) is defined as 1.
 *
 * @param x Input array
 * @return Array with same shape and dtype as x
 */
Array sinc(const Array &x);

/**
 * @brief 1D convolution of two arrays.
 *
 * Uses FFT-based convolution for efficiency.
 *
 * @param a First input array (1D)
 * @param v Second input array (1D)
 * @param mode Convolution mode: "full", "same", "valid"
 * @return Convolved array
 */
Array convolve(const Array &a, const Array &v,
               const std::string &mode = "full");

} // namespace ins