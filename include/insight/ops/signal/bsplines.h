// insight/ops/signal/bsplines.h
#pragma once
#include "insight/core/array.h"

namespace ins {
namespace signal {

/**
 * @brief Gaussian approximation to B-spline basis function of order n.
 *
 * gauss_spline(x, n) = 1/sqrt(2*pi*sigma^2) * exp(-x^2 / (2*sigma^2))
 * where sigma^2 = (n+1)/12
 *
 * @param x Input array
 * @param n Spline order (must be >= 0)
 * @return Array with same shape as x
 */
Array gauss_spline(const Array &x, int n);

/**
 * @brief Cubic B-spline (equivalent to bspline(x, 3)).
 *
 * Piecewise polynomial:
 *   |x| < 1:   2/3 - |x|^2*(2-|x|)/2
 *   1 <= |x| < 2: (2-|x|)^3 / 6
 *   |x| >= 2:  0
 *
 * @param x Input array
 * @return Array with same shape as x
 */
Array cubic(const Array &x);

/**
 * @brief Quadratic B-spline (equivalent to bspline(x, 2)).
 *
 * Piecewise polynomial:
 *   |x| < 0.5:   0.75 - |x|^2
 *   0.5 <= |x| < 1.5: 0.5*(|x|-1.5)^2
 *   |x| >= 1.5:  0
 *
 * @param x Input array
 * @return Array with same shape as x
 */
Array quadratic(const Array &x);

} // namespace signal
} // namespace ins
