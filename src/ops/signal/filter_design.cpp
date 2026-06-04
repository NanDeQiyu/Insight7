// src/ops/signal/filter_design.cpp
#include "insight/ops/signal/filter_design.h"
#include "insight/core/exception.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/reduction.h"
#include "insight/ops/signal/windows.h"
#include "insight/ops/unary.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ins {
namespace signal {

// ============================================================================
// kaiser_beta
// ============================================================================

double kaiser_beta(double a) {
  if (a > 50.0) {
    return 0.1102 * (a - 8.7);
  } else if (a > 21.0) {
    return 0.5842 * std::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0);
  }
  return 0.0;
}

// ============================================================================
// kaiser_atten
// ============================================================================

double kaiser_atten(int64_t numtaps, double width) {
  INS_CHECK(numtaps >= 1, "kaiser_atten: numtaps must be >= 1");
  INS_CHECK(width > 0.0 && width <= 1.0,
            "kaiser_atten: width must be in (0, 1]");
  return 2.285 * (numtaps - 1) * M_PI * width + 7.95;
}

// ============================================================================
// firwin — FIR filter design using windowed sinc
// ============================================================================

namespace {

bool parse_pass_zero(const std::string &pz) {
  if (pz == "lowpass" || pz == "bandstop")
    return true;
  if (pz == "highpass" || pz == "bandpass")
    return false;
  INS_CHECK(false,
            "firwin: pass_zero must be 'lowpass', 'highpass', "
            "'bandpass', or 'bandstop', got '",
            pz, "'");
  return true;
}

Array firwin_cpu(int64_t numtaps, const std::vector<double> &cutoff,
                 const std::string &window, bool pass_zero, bool scale) {
  Place cpu = CPUPlace();
  int64_t n = numtaps;
  double alpha = 0.5 * (n - 1);

  Array m_arr = arange(0.0, static_cast<double>(n), 1.0, DType::F64, cpu);
  Array m_minus_alpha = sub(m_arr, full({n}, alpha, DType::F64, cpu));

  Array result;
  if (pass_zero) {
    result = ones({n}, DType::F64, cpu);
  } else {
    result = zeros({n}, DType::F64, cpu);
  }

  for (size_t i = 0; i < cutoff.size(); ++i) {
    double c = cutoff[i];
    INS_CHECK(c > 0.0 && c < 1.0, "firwin: cutoff must be in (0, 1), got ", c);

    Array arg = mul(m_minus_alpha, full({n}, c, DType::F64, cpu));
    Array pi_arg = mul(arg, full({n}, M_PI, DType::F64, cpu));
    Array sin_val = sin(pi_arg);
    Array eps = full({n}, 1e-12, DType::F64, cpu);
    Array abs_pi_arg = abs(pi_arg);
    Array is_zero = less_equal(abs_pi_arg, eps);
    Array sinc_val = div(sin_val, pi_arg);
    Array one = full({n}, 1.0, DType::F64, cpu);
    sinc_val = where(is_zero, one, sinc_val);
    Array h = mul(sinc_val, full({n}, c, DType::F64, cpu));

    if (i % 2 == 0) {
      result = pass_zero ? sub(result, h) : add(result, h);
    } else {
      result = pass_zero ? add(result, h) : sub(result, h);
    }
  }

  // Apply window
  Array win = get_window(window, numtaps, false);
  if (win.place().kind() != DeviceKind::CPU)
    win = win.to(cpu);
  result = mul(result, win);

  // Scale using composite ops
  if (scale) {
    // Compute peak gain by evaluating DFT across the passband.
    // For a symmetric (Type I/II) FIR filter:
    //   H(f) = sum_k h[k] * cos(pi * k * f)  (Type I, odd length)
    // We want to find max |H(f)| in the passband and normalize to 1.
    //
    // Lowpass/bandstop (pass_zero=true): passband includes DC, omega=0.
    // Highpass/bandpass (pass_zero=false): passband excludes DC.
    double peak_gain = 0.0;

    if (pass_zero && cutoff.size() < 2) {
      // Lowpass: gain at DC = sum(h)
      peak_gain = sum(result).item<double>();
    } else if (!pass_zero && cutoff.size() < 2) {
      // Highpass: gain at Nyquist = alternating sum h[k]*(-1)^k
      // This is the DTFT evaluated at omega = pi.
      const double *h_data = result.data<double>();
      double alt_sum = 0.0;
      for (int64_t k = 0; k < n; ++k) {
        alt_sum += h_data[k] * ((k % 2 == 0) ? 1.0 : -1.0);
      }
      peak_gain = alt_sum;
    } else {
      // Bandpass/bandstop: evaluate DFT across frequency grid
      // and find the peak magnitude.
      int64_t nfreq = std::max((int64_t)256, 4 * n);
      double df = M_PI / nfreq;

      // Determine frequency search range
      double f_lo = 0.0, f_hi = M_PI;
      if (!pass_zero) {
        if (cutoff.size() >= 2) {
          // Bandpass: search between first and last cutoff
          f_lo = M_PI * cutoff.front() * 0.5;
          f_hi = M_PI * cutoff.back() * 1.5;
          f_hi = std::min(f_hi, M_PI);
        } else {
          // Highpass: search from cutoff to Nyquist (inclusive)
          f_lo = M_PI * cutoff[0];
          f_hi = M_PI;
        }
      } else {
        // Bandstop: search from DC to first cutoff
        f_hi = M_PI * cutoff.front();
      }

      const double *h_data = result.data<double>();
      for (int64_t fi = 0; fi <= nfreq; ++fi) {
        double freq = f_lo + (f_hi - f_lo) * fi / nfreq;
        double re = 0.0;
        for (int64_t k = 0; k < n; ++k) {
          re += h_data[k] * std::cos(k * freq);
        }
        double mag = std::abs(re);
        if (mag > peak_gain)
          peak_gain = mag;
      }
    }

    if (std::abs(peak_gain) > 1e-15) {
      result = div(result, full(result.shape(), peak_gain, DType::F64, cpu));
    }
  }

  return result;
}

} // anonymous namespace

Array firwin(int64_t numtaps, const std::vector<double> &cutoff,
             const std::string &window, const std::string &pass_zero_str,
             bool scale) {
  INS_CHECK(numtaps >= 1, "firwin: numtaps must be >= 1");
  INS_CHECK(!cutoff.empty(), "firwin: at least one cutoff frequency required");
  bool pz = parse_pass_zero(pass_zero_str);
  return firwin_cpu(numtaps, cutoff, window, pz, scale);
}

// ============================================================================
// firwin2 — FIR filter design from frequency-magnitude pairs
// ============================================================================

Array firwin2(int64_t numtaps, const std::vector<double> &freq,
              const std::vector<double> &gain, int64_t nfreqs,
              const std::string &window, bool antisymmetric) {
  INS_CHECK(numtaps >= 1, "firwin2: numtaps must be >= 1");
  INS_CHECK(freq.size() >= 2, "firwin2: need at least 2 frequency points");
  INS_CHECK(freq.size() == gain.size(),
            "firwin2: freq and gain must have same length");
  INS_CHECK(std::abs(freq.front()) < 1e-10,
            "firwin2: first frequency must be 0");
  INS_CHECK(std::abs(freq.back() - 1.0) < 1e-10,
            "firwin2: last frequency must be 1");

  Place cpu = CPUPlace();

  if (nfreqs == 0) {
    nfreqs = 1;
    while (nfreqs < numtaps)
      nfreqs *= 2;
  }

  // Interpolate the desired frequency response
  int64_t n_half = nfreqs / 2 + 1;
  std::vector<double> response(n_half);
  double df = 1.0 / (n_half - 1);
  size_t j = 0;
  for (int64_t i = 0; i < n_half; ++i) {
    double f = i * df;
    while (j + 1 < freq.size() - 1 && freq[j + 1] < f)
      ++j;
    double f0 = freq[j], f1 = freq[j + 1];
    double g0 = gain[j], g1 = gain[j + 1];
    double t = (f1 > f0) ? (f - f0) / (f1 - f0) : 0.0;
    response[i] = g0 + t * (g1 - g0);
  }

  // Build full complex response
  int64_t fft_len = (n_half - 1) * 2;
  std::vector<std::complex<double>> full_response(fft_len);

  if (antisymmetric) {
    double phase_shift = M_PI * (numtaps - 1) / (2.0 * fft_len);
    for (int64_t k = 0; k < fft_len; ++k) {
      if (k < n_half) {
        full_response[k] =
            response[k] * std::exp(std::complex<double>(0, -k * phase_shift));
      } else {
        full_response[k] = std::conj(full_response[fft_len - k]);
      }
    }
  } else {
    for (int64_t k = 0; k < fft_len; ++k) {
      if (k < n_half) {
        full_response[k] = response[k];
      } else {
        full_response[k] = response[fft_len - k];
      }
    }
  }

  // IFFT
  Array resp_arr = to_array(full_response, DType::C64, cpu);
  Array h_full = fft::ifft(resp_arr);

  // Extract real part using framework API
  Array h_real = real(h_full);
  Array h = slice(h_real, {0}, {0}, {static_cast<int>(numtaps)});
  if (h.dtype() != DType::F64)
    h = h.to(DType::F64);

  // Apply window
  Array win = get_window(window, numtaps, false);
  if (win.place().kind() != DeviceKind::CPU)
    win = win.to(cpu);
  h = mul(h, win);

  // Scale
  if (gain[0] != 0.0) {
    double dc = sum(h).item<double>();
    if (std::abs(dc) > 1e-15) {
      h = div(h, full(h.shape(), dc / gain[0], DType::F64, cpu));
    }
  }

  return h;
}

// ============================================================================
// cmplx_sort — Sort complex values by magnitude
// ============================================================================

Array cmplx_sort(const Array &p) {
  INS_CHECK(p.defined(), "cmplx_sort: input is undefined");

  Place cpu = CPUPlace();
  Array p_cpu = (p.place().kind() == DeviceKind::CPU) ? p : p.to(cpu);

  // Sort by magnitude — compute |z|^2 = re^2 + im^2 to stay in real domain
  // abs() on complex may return complex in some implementations
  if (is_complex(p_cpu)) {
    Array re = real(p_cpu);
    Array im = imag(p_cpu);
    Array mag_sq = add(square(re), square(im));
    Array sort_idx = argsort(mag_sq, 0, false);
    Array sorted = take(p_cpu, sort_idx);
    return sorted;
  }

  // Real values: sort by absolute value
  Array mag_sq = square(p_cpu);
  Array sort_idx = argsort(mag_sq, 0, false);
  Array sorted = take(p_cpu, sort_idx);
  return sorted;
}

} // namespace signal
} // namespace ins
