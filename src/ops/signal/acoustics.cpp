// src/ops/signal/acoustics.cpp
#include "insight/ops/signal/acoustics.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/unary.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ins {
namespace signal {

// ============================================================================
// mel2hz / hz2mel
// ============================================================================

// HTK formula: mel = 2595 * log10(1 + hz/700)
// Inverse: hz = 700 * (10^(mel/2595) - 1)

Array mel2hz(const Array &mels) {
  INS_CHECK(mels.defined(), "mel2hz: input is undefined");

  // 700 * (10^(mels/2595) - 1)
  // = 700 * (exp(mels/2595 * ln(10)) - 1)
  double scale = std::log(10.0) / 2595.0;
  Array exp_val = exp(mul(mels, full(mels.shape(), scale, mels.dtype())));
  Array result = mul(sub(exp_val, full(exp_val.shape(), 1.0, exp_val.dtype())),
                     full(exp_val.shape(), 700.0, exp_val.dtype()));
  return result;
}

Array hz2mel(const Array &hz) {
  INS_CHECK(hz.defined(), "hz2mel: input is undefined");

  // 2595 * log10(1 + hz/700)
  // = 2595 * log(1 + hz/700) / log(10)
  Array hz_over_700 = div(hz, full(hz.shape(), 700.0, hz.dtype()));
  Array one_plus =
      add(hz_over_700, full(hz_over_700.shape(), 1.0, hz_over_700.dtype()));
  Array log_val = log(one_plus);
  double scale = 2595.0 / std::log(10.0);
  return mul(log_val, full(log_val.shape(), scale, log_val.dtype()));
}

// ============================================================================
// mel_frequencies
// ============================================================================

Array mel_frequencies(int64_t n_mels, double f_low, double f_high) {
  INS_CHECK(n_mels >= 1, "mel_frequencies: n_mels must be >= 1");

  Place cpu = CPUPlace();

  // Compute mel endpoints
  double m_low = 2595.0 * std::log10(1.0 + f_low / 700.0);
  double m_high = 2595.0 * std::log10(1.0 + f_high / 700.0);

  // Linearly spaced mel values
  std::vector<double> mels(n_mels);
  if (n_mels == 1) {
    mels[0] = (m_low + m_high) / 2.0;
  } else {
    double dm = (m_high - m_low) / (n_mels - 1);
    for (int64_t i = 0; i < n_mels; ++i) {
      mels[i] = m_low + i * dm;
    }
  }

  // Convert back to Hz
  std::vector<double> hz(n_mels);
  for (int64_t i = 0; i < n_mels; ++i) {
    hz[i] = 700.0 * (std::pow(10.0, mels[i] / 2595.0) - 1.0);
  }

  return to_array(hz, DType::F64, cpu);
}

// ============================================================================
// hz2bark / bark2hz
// ============================================================================

// Traunmüller (1990) formula:
// bark = 26.81 / (1 + 1960/hz) - 0.53
// Inverse: hz = 1960 / (26.81/(bark + 0.53) - 1)

Array hz2bark(const Array &hz) {
  INS_CHECK(hz.defined(), "hz2bark: input is undefined");

  // bark = 26.81 / (1 + 1960/hz) - 0.53
  Array inv_hz = div(full(hz.shape(), 1960.0, hz.dtype()), hz);
  Array denom = add(inv_hz, full(inv_hz.shape(), 1.0, inv_hz.dtype()));
  Array bark = sub(div(full(denom.shape(), 26.81, denom.dtype()), denom),
                   full(denom.shape(), 0.53, denom.dtype()));
  return bark;
}

Array bark2hz(const Array &bark) {
  INS_CHECK(bark.defined(), "bark2hz: input is undefined");

  // hz = 1960 / (26.81/(bark + 0.53) - 1)
  Array bark_shifted = add(bark, full(bark.shape(), 0.53, bark.dtype()));
  Array ratio = div(full(bark_shifted.shape(), 26.81, bark_shifted.dtype()),
                    bark_shifted);
  Array denom = sub(ratio, full(ratio.shape(), 1.0, ratio.dtype()));
  Array hz = div(full(denom.shape(), 1960.0, denom.dtype()), denom);
  return hz;
}

} // namespace signal
} // namespace ins
