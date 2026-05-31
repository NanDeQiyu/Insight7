// src/ops/signal/spectral_analysis.cpp
#include "insight/ops/signal/spectral_analysis.h"
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

namespace {

// Detrend a segment (remove mean)
Array detrend_segment(const Array &seg, const std::string &method) {
  if (method == "none" || method.empty())
    return seg;
  if (method == "constant") {
    Array m = mean(seg);
    return sub(seg, m);
  }
  return seg;
}

// Get window array from name
Array get_window_for_spectral(const std::string &window, int64_t nperseg) {
  return get_window(window, nperseg, false); // symmetric
}

// Compute one-sided FFT length
int64_t onesided_fft_len(int64_t n, bool onesided) {
  if (onesided)
    return n / 2 + 1;
  return n;
}

// Compute frequency array
Array fft_freqs(int64_t nfft, double fs, bool onesided) {
  int64_t n = onesided ? (nfft / 2 + 1) : nfft;
  std::vector<double> freqs(n);
  double df = fs / nfft;
  for (int64_t i = 0; i < n; ++i) {
    freqs[i] = i * df;
  }
  return to_array(freqs, DType::F64, CPUPlace());
}

// Scale PSD
double psd_scale(int64_t nfft, double fs, const Array &window,
                 const std::string &scaling, bool onesided) {
  double scale = 1.0;
  if (scaling == "density") {
    scale = fs;
  } else { // "spectrum"
    scale = 1.0;
  }

  // Window normalization: 1/sum(w^2)
  const double *w = window.data<double>();
  double win_sum_sq = 0.0;
  for (int64_t i = 0; i < window.numel(); ++i) {
    win_sum_sq += w[i] * w[i];
  }
  scale /= win_sum_sq;

  return scale;
}

} // anonymous namespace

// ============================================================================
// csd — Cross Power Spectral Density
// ============================================================================

SpectralResult csd(const Array &x, const Array &y, double fs,
                   const std::string &window, int64_t nperseg, int64_t noverlap,
                   int64_t nfft, const std::string &detrend,
                   bool return_onesided, const std::string &scaling) {
  INS_CHECK(x.defined() && y.defined(), "csd: inputs are undefined");

  Place cpu = CPUPlace();
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);
  Array y_cpu = (y.place().kind() == DeviceKind::CPU) ? y : y.to(cpu);

  if (nfft == 0)
    nfft = nperseg;
  if (noverlap == 0)
    noverlap = nperseg / 2;

  int64_t n = x_cpu.numel();
  int64_t step = nperseg - noverlap;

  // Get window
  Array win = get_window_for_spectral(window, nperseg);

  // Compute number of segments
  int64_t n_segments = 0;
  if (n >= nperseg) {
    n_segments = (n - nperseg) / step + 1;
  }
  INS_CHECK(n_segments >= 1, "csd: signal too short for given nperseg");

  int64_t freq_len = onesided_fft_len(nfft, return_onesided);

  // Accumulate PSD
  std::vector<double> Pxx_accum(freq_len, 0.0);

  const double *x_data = x_cpu.data<double>();
  const double *y_data = y_cpu.data<double>();
  const double *w_data = win.data<double>();

  for (int64_t seg = 0; seg < n_segments; ++seg) {
    int64_t start = seg * step;

    // Extract and window the segment
    std::vector<std::complex<double>> x_seg(nperseg);
    std::vector<std::complex<double>> y_seg(nperseg);
    for (int64_t i = 0; i < nperseg; ++i) {
      double xi = x_data[start + i];
      double yi = y_data[start + i];
      if (detrend == "constant") {
        // Will detrend below
      }
      x_seg[i] = {xi * w_data[i], 0.0};
      y_seg[i] = {yi * w_data[i], 0.0};
    }

    // Detrend (subtract mean from windowed segment)
    if (detrend == "constant") {
      double x_mean = 0.0, y_mean = 0.0;
      for (int64_t i = 0; i < nperseg; ++i) {
        x_mean += x_seg[i].real();
        y_mean += y_seg[i].real();
      }
      x_mean /= nperseg;
      y_mean /= nperseg;
      for (int64_t i = 0; i < nperseg; ++i) {
        x_seg[i] = {x_seg[i].real() - x_mean, 0.0};
        y_seg[i] = {y_seg[i].real() - y_mean, 0.0};
      }
    }

    // FFT
    Array x_arr = to_array(x_seg, DType::C64, cpu);
    Array y_arr = to_array(y_seg, DType::C64, cpu);
    Array Xf = fft::fft(x_arr, nfft);
    Array Yf = fft::fft(y_arr, nfft);

    const std::complex<double> *X_data =
        reinterpret_cast<const std::complex<double> *>(Xf.data<char>());
    const std::complex<double> *Y_data =
        reinterpret_cast<const std::complex<double> *>(Yf.data<char>());

    // Pxy = conj(X) * Y
    for (int64_t i = 0; i < freq_len; ++i) {
      std::complex<double> pxy = std::conj(X_data[i]) * Y_data[i];
      Pxx_accum[i] += pxy.real();
    }
  }

  // Average and scale
  double scale = psd_scale(nfft, fs, win, scaling, return_onesided);
  for (int64_t i = 0; i < freq_len; ++i) {
    Pxx_accum[i] = (Pxx_accum[i] / n_segments) * scale;
  }

  // For one-sided, double the non-DC and non-Nyquist components
  if (return_onesided && nfft > 1) {
    for (int64_t i = 1; i < freq_len - 1; ++i) {
      Pxx_accum[i] *= 2.0;
    }
  }

  Array f = fft_freqs(nfft, fs, return_onesided);
  Array Pxx = to_array(Pxx_accum, DType::F64, cpu);

  return {f, Pxx};
}

// ============================================================================
// welch — Power Spectral Density
// ============================================================================

SpectralResult welch(const Array &x, double fs, const std::string &window,
                     int64_t nperseg, int64_t noverlap, int64_t nfft,
                     const std::string &detrend, bool return_onesided,
                     const std::string &scaling) {
  // Welch = auto-csd, return real part
  SpectralResult result = csd(x, x, fs, window, nperseg, noverlap, nfft,
                              detrend, return_onesided, scaling);
  // PSD of real signal should be real (imaginary part ~ 0)
  return result;
}

// ============================================================================
// periodogram — Convenience wrapper
// ============================================================================

SpectralResult periodogram(const Array &x, double fs, const std::string &window,
                           int64_t nfft, const std::string &detrend,
                           bool return_onesided, const std::string &scaling) {
  return welch(x, fs, window, 256, 0, nfft, detrend, return_onesided, scaling);
}

// ============================================================================
// coherence
// ============================================================================

SpectralResult coherence(const Array &x, const Array &y, double fs,
                         const std::string &window, int64_t nperseg,
                         int64_t noverlap, int64_t nfft,
                         const std::string &detrend) {
  SpectralResult Pxx = welch(x, fs, window, nperseg, noverlap, nfft, detrend);
  SpectralResult Pyy = welch(y, fs, window, nperseg, noverlap, nfft, detrend);
  SpectralResult Pxy = csd(x, y, fs, window, nperseg, noverlap, nfft, detrend);

  // Cxy = |Pxy|^2 / (Pxx * Pyy)
  Place cpu = CPUPlace();
  const double *pxx = Pxx.Pxx.data<double>();
  const double *pyy = Pyy.Pxx.data<double>();
  const double *pxy = Pxy.Pxx.data<double>();

  int64_t n = Pxx.Pxx.numel();
  std::vector<double> cxy(n);
  for (int64_t i = 0; i < n; ++i) {
    double denom = pxx[i] * pyy[i];
    cxy[i] = (denom > 1e-30) ? (pxy[i] * pxy[i]) / denom : 0.0;
  }

  return {Pxx.f, to_array(cxy, DType::F64, cpu)};
}

// ============================================================================
// spectrogram
// ============================================================================

SpectrogramResult spectrogram(const Array &x, double fs,
                              const std::string &window, int64_t nperseg,
                              int64_t noverlap, int64_t nfft,
                              const std::string &detrend, bool return_onesided,
                              const std::string &mode) {
  Place cpu = CPUPlace();
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);

  if (nfft == 0)
    nfft = nperseg;
  if (noverlap == 0)
    noverlap = nperseg / 8; // spectrogram default

  int64_t n = x_cpu.numel();
  int64_t step = nperseg - noverlap;
  int64_t n_segments = (n >= nperseg) ? ((n - nperseg) / step + 1) : 0;
  INS_CHECK(n_segments >= 1, "spectrogram: signal too short");

  int64_t freq_len = onesided_fft_len(nfft, return_onesided);

  Array win = get_window_for_spectral(window, nperseg);
  const double *x_data = x_cpu.data<double>();
  const double *w_data = win.data<double>();

  // Compute time array
  std::vector<double> t_arr(n_segments);
  for (int64_t i = 0; i < n_segments; ++i) {
    t_arr[i] = (i * step + nperseg / 2.0) / fs;
  }

  // Compute spectrogram
  std::vector<double> Sxx(freq_len * n_segments);

  for (int64_t seg = 0; seg < n_segments; ++seg) {
    int64_t start = seg * step;

    // Window and detrend
    std::vector<std::complex<double>> seg_data(nperseg);
    double seg_mean = 0.0;
    if (detrend == "constant") {
      for (int64_t i = 0; i < nperseg; ++i)
        seg_mean += x_data[start + i];
      seg_mean /= nperseg;
    }
    for (int64_t i = 0; i < nperseg; ++i) {
      seg_data[i] = {(x_data[start + i] - seg_mean) * w_data[i], 0.0};
    }

    // FFT
    Array seg_arr = to_array(seg_data, DType::C64, cpu);
    Array Xf = fft::fft(seg_arr, nfft);
    const std::complex<double> *X_data =
        reinterpret_cast<const std::complex<double> *>(Xf.data<char>());

    if (mode == "psd" || mode == "complex") {
      for (int64_t i = 0; i < freq_len; ++i) {
        if (mode == "complex") {
          Sxx[seg * freq_len + i] = X_data[i].real(); // store magnitude
        } else {
          Sxx[seg * freq_len + i] = std::norm(X_data[i]);
        }
      }
    } else if (mode == "magnitude") {
      for (int64_t i = 0; i < freq_len; ++i) {
        Sxx[seg * freq_len + i] = std::abs(X_data[i]);
      }
    }
  }

  // Scale PSD
  if (mode == "psd") {
    double scale = psd_scale(nfft, fs, win, "density", return_onesided);
    for (auto &v : Sxx)
      v *= scale;
    if (return_onesided && nfft > 1) {
      for (int64_t seg = 0; seg < n_segments; ++seg) {
        for (int64_t i = 1; i < freq_len - 1; ++i) {
          Sxx[seg * freq_len + i] *= 2.0;
        }
      }
    }
  }

  Array f = fft_freqs(nfft, fs, return_onesided);
  Array t = to_array(t_arr, DType::F64, cpu);
  Array S = to_array(Sxx, Shape({n_segments, freq_len}), DType::F64, cpu);

  return {f, t, S};
}

// ============================================================================
// stft
// ============================================================================

SpectrogramResult stft(const Array &x, double fs, const std::string &window,
                       int64_t nperseg, int64_t noverlap, int64_t nfft) {
  return spectrogram(x, fs, window, nperseg, noverlap, nfft, "none", true,
                     "complex");
}

// ============================================================================
// vectorstrength
// ============================================================================

std::pair<double, double> vectorstrength(const Array &events, double period) {
  INS_CHECK(events.defined(), "vectorstrength: events is undefined");
  INS_CHECK(period > 0, "vectorstrength: period must be > 0");

  Place cpu = CPUPlace();
  Array events_cpu =
      (events.place().kind() == DeviceKind::CPU) ? events : events.to(cpu);

  int64_t n = events_cpu.numel();
  const double *ev_data = events_cpu.data<double>();

  // Convert to phasors: exp(2j*pi*events/period)
  double real_sum = 0.0, imag_sum = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    double phase = 2.0 * M_PI * ev_data[i] / period;
    real_sum += std::cos(phase);
    imag_sum += std::sin(phase);
  }

  real_sum /= n;
  imag_sum /= n;

  double strength = std::sqrt(real_sum * real_sum + imag_sum * imag_sum);
  double angle = std::atan2(imag_sum, real_sum);

  return {strength, angle};
}

} // namespace signal
} // namespace ins
