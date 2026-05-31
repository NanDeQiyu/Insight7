// src/ops/signal/filter_design.cpp
#include "insight/ops/signal/filter_design.h"
#include "insight/core/exception.h"
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

// Compute windowed sinc FIR filter on CPU (scalar reads require CPU arrays)
Array firwin_cpu(int64_t numtaps, const std::vector<double> &cutoff,
                 const std::string &window, bool pass_zero, bool scale) {
  Place cpu = CPUPlace();
  int64_t n = numtaps;
  double alpha = 0.5 * (n - 1);

  // Build m indices on CPU
  Array m_arr = arange(0.0, static_cast<double>(n), 1.0, DType::F64, cpu);
  Array m_minus_alpha = sub(m_arr, full({n}, alpha, DType::F64, cpu));

  // Start with zeros (bandpass) or ones (bandstop/lowpass)
  Array result;
  if (pass_zero) {
    result = ones({n}, DType::F64, cpu);
  } else {
    result = zeros({n}, DType::F64, cpu);
  }

  // Add/subtract ideal lowpass responses for each cutoff
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
      if (pass_zero)
        result = sub(result, h);
      else
        result = add(result, h);
    } else {
      if (pass_zero)
        result = add(result, h);
      else
        result = sub(result, h);
    }
  }

  // Apply window — build on CPU to avoid device mismatch
  // Use get_window with CPU transfer
  Array win = get_window(window, numtaps, false);
  if (win.place().kind() != DeviceKind::CPU) {
    win = win.to(cpu);
  }
  result = mul(result, win);

  // Scale
  if (scale) {
    double scale_factor = 0.0;
    const double *hd = result.data<double>();
    if (pass_zero) {
      for (int64_t k = 0; k < result.numel(); ++k)
        scale_factor += hd[k];
    } else {
      for (int64_t k = 0; k < result.numel(); ++k) {
        scale_factor += hd[k] * (k % 2 == 0 ? 1.0 : -1.0);
      }
    }
    if (std::abs(scale_factor) > 1e-15) {
      result = div(result, full(result.shape(), scale_factor, DType::F64, cpu));
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
    double f0 = freq[j];
    double f1 = freq[j + 1];
    double g0 = gain[j];
    double g1 = gain[j + 1];
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
        int64_t mirror = fft_len - k;
        full_response[k] = std::conj(full_response[mirror]);
      }
    }
  } else {
    for (int64_t k = 0; k < fft_len; ++k) {
      if (k < n_half) {
        full_response[k] = response[k];
      } else {
        int64_t mirror = fft_len - k;
        full_response[k] = response[mirror];
      }
    }
  }

  // IFFT (on CPU)
  Array resp_arr = to_array(full_response, DType::C64, cpu);
  Array h_full = fft::ifft(resp_arr);

  // Extract real part
  std::vector<double> h_vals(numtaps);
  if (h_full.dtype() == DType::C64) {
    const std::complex<double> *data =
        reinterpret_cast<const std::complex<double> *>(h_full.data<char>());
    for (int64_t i = 0; i < numtaps; ++i) {
      h_vals[i] = data[i].real();
    }
  } else {
    const double *data = h_full.data<double>();
    for (int64_t i = 0; i < numtaps; ++i) {
      h_vals[i] = data[i];
    }
  }

  Array h = to_array(h_vals, DType::F64, cpu);

  // Apply window (force CPU)
  Array win = get_window(window, numtaps, false);
  if (win.place().kind() != DeviceKind::CPU) {
    win = win.to(cpu);
  }
  h = mul(h, win);

  // Scale
  if (gain[0] != 0.0) {
    const double *hd = h.data<double>();
    double dc = 0.0;
    for (int64_t i = 0; i < h.numel(); ++i)
      dc += hd[i];
    if (std::abs(dc) > 1e-15) {
      h = div(h, full(h.shape(), dc / gain[0], DType::F64, cpu));
    }
  }

  return h;
}

// ============================================================================
// cmplx_sort
// ============================================================================

Array cmplx_sort(const Array &p) {
  INS_CHECK(p.defined(), "cmplx_sort: input is undefined");

  Place cpu = CPUPlace();

  // Transfer to CPU if needed (scalar sort is inherently serial)
  Array p_cpu = p;
  if (p.place().kind() != DeviceKind::CPU) {
    p_cpu = p.to(cpu);
  }

  int64_t n = p_cpu.numel();

  if (p_cpu.dtype() == DType::C64) {
    const std::complex<double> *data =
        reinterpret_cast<const std::complex<double> *>(p_cpu.data<char>());
    std::vector<std::complex<double>> vals(data, data + n);
    std::sort(vals.begin(), vals.end(),
              [](const std::complex<double> &a, const std::complex<double> &b) {
                return std::abs(a) < std::abs(b);
              });
    return to_array(vals, DType::C64, cpu);
  } else if (p_cpu.dtype() == DType::C32) {
    const std::complex<float> *data =
        reinterpret_cast<const std::complex<float> *>(p_cpu.data<char>());
    std::vector<std::complex<float>> vals(data, data + n);
    std::sort(vals.begin(), vals.end(),
              [](const std::complex<float> &a, const std::complex<float> &b) {
                return std::abs(a) < std::abs(b);
              });
    return to_array(vals, DType::C32, cpu);
  } else if (p_cpu.dtype() == DType::F64) {
    const double *data = p_cpu.data<double>();
    std::vector<double> vals(data, data + n);
    std::sort(vals.begin(), vals.end(),
              [](double a, double b) { return std::abs(a) < std::abs(b); });
    return to_array(vals, DType::F64, cpu);
  } else if (p_cpu.dtype() == DType::F32) {
    const float *data = p_cpu.data<float>();
    std::vector<float> vals(data, data + n);
    std::sort(vals.begin(), vals.end(),
              [](float a, float b) { return std::abs(a) < std::abs(b); });
    return to_array(vals, DType::F32, cpu);
  }

  INS_CHECK(false, "cmplx_sort: unsupported dtype");
  return Array();
}

} // namespace signal
} // namespace ins
