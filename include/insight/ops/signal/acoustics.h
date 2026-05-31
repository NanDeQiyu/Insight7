// insight/ops/signal/acoustics.h
#pragma once
#include "insight/core/array.h"

namespace ins {
namespace signal {

/// @brief Convert mel frequencies to Hz.
/// @param mels Input mel frequency (scalar or array)
/// @return Hz frequency
Array mel2hz(const Array &mels);

/// @brief Convert Hz frequencies to mel scale.
/// @param hz Input Hz frequency (scalar or array)
/// @return Mel frequency
Array hz2mel(const Array &hz);

/// @brief Compute mel frequencies equally spaced between f_low and f_high.
/// @param n_mels Number of mel bins
/// @param f_low Lowest frequency in Hz. Default: 0
/// @param f_high Highest frequency in Hz. Default: 11000
/// @return Array of n_mels mel frequencies in Hz
Array mel_frequencies(int64_t n_mels, double f_low = 0.0,
                      double f_high = 11000.0);

/// @brief Convert Hz to Bark scale.
/// @param hz Input Hz frequency (scalar or array)
/// @return Bark frequency
Array hz2bark(const Array &hz);

/// @brief Convert Bark scale to Hz.
/// @param bark Input Bark frequency (scalar or array)
/// @return Hz frequency
Array bark2hz(const Array &bark);

} // namespace signal
} // namespace ins
