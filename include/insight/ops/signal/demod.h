// insight/ops/signal/demod.h
#pragma once
#include "insight/core/array.h"

namespace ins {
namespace signal {

/**
 * @brief FM demodulation of a complex-valued signal.
 *
 * Computes diff(unwrap(angle(x))) along the specified axis.
 * Input must be complex-valued (C32 or C64).
 *
 * @param x Complex-valued input signal
 * @param axis Axis along which to demodulate. Default: -1
 * @return Real-valued demodulated signal (1 shorter along axis)
 */
Array fm_demod(const Array &x, int axis = -1);

} // namespace signal
} // namespace ins
