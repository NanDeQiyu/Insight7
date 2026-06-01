// src/ops/signal/spectral_analysis.cpp
#include "insight/ops/signal/spectral_analysis.h"
#include "insight/core/exception.h"
#include "insight/core/op_registry.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
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

Array detrend_segment(const Array &seg, const std::string &method) {
  if (method == "none" || method.empty())
    return seg;
  if (method == "constant") {
    Array m = mean(seg);
    return sub(seg, m);
  }
  return seg;
}

Array get_window_for_spectral(const std::string &window, int64_t nperseg) {
  return get_window(window, nperseg, false);
}

int64_t onesided_fft_len(int64_t n, bool onesided) {
  return onesided ? (n / 2 + 1) : n;
}

Array fft_freqs(int64_t nfft, double fs, bool onesided) {
  int64_t n = onesided ? (nfft / 2 + 1) : nfft;
  std::vector<double> freqs(n);
  double df = fs / nfft;
  for (int64_t i = 0; i < n; ++i) {
    freqs[i] = i * df;
  }
  return to_array(freqs, DType::F64, CPUPlace());
}

double compute_psd_scale(int64_t nfft, double fs, const Array &window,
                         const std::string &scaling, bool onesided) {
  double scale = (scaling == "density") ? fs : 1.0;

  // Window normalization: 1/sum(w^2)
  Array win_sq = square(window);
  double win_sum_sq = sum(win_sq).item<double>();
  scale /= win_sum_sq;

  return scale;
}

// Process one segment: window, detrend, FFT, return cross-spectrum
Array process_segment(const Array &x_seg, const Array &y_seg, const Array &win,
                      int64_t nfft, const std::string &detrend_method) {
  Place cpu = CPUPlace();

  // Window the segments
  Array x_win = mul(x_seg, win);
  Array y_win = mul(y_seg, win);

  // Detrend
  if (detrend_method == "constant") {
    x_win = sub(x_win, mean(x_win));
    y_win = sub(y_win, mean(y_win));
  }

  // Convert to complex for FFT
  DType cdtype = DType::C64;
  Array x_cpx = to_complex(x_win);
  Array y_cpx = to_complex(y_win);

  // FFT
  Array Xf = fft::fft(x_cpx, nfft);
  Array Yf = fft::fft(y_cpx, nfft);

  // Pxy = conj(X) * Y
  Array Xf_conj = conj(Xf);
  return mul(Xf_conj, Yf);
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

  Array win = get_window_for_spectral(window, nperseg);

  int64_t n_segments = (n >= nperseg) ? ((n - nperseg) / step + 1) : 0;
  INS_CHECK(n_segments >= 1, "csd: signal too short for given nperseg");

  int64_t freq_len = onesided_fft_len(nfft, return_onesided);

  // Accumulate cross-spectrum using Array composites
  Array Pxx_sum = zeros({freq_len}, DType::F64, cpu);

  for (int64_t seg = 0; seg < n_segments; ++seg) {
    int64_t start = seg * step;

    // Extract segments
    Array x_seg = slice(x_cpu, {0}, {start}, {start + nperseg});
    Array y_seg = slice(y_cpu, {0}, {start}, {start + nperseg});

    // Window and compute cross-spectrum
    if (win.dtype() != x_seg.dtype())
      win = win.to(x_seg.dtype());
    Array Pxy = process_segment(x_seg, y_seg, win, nfft, detrend);

    // Accumulate cross-spectrum using composite ops (no per-element extraction)
    Array Pxy_cpu = (Pxy.place().kind() == DeviceKind::CPU) ? Pxy : Pxy.to(cpu);
    Array Pxy_real_full = real(Pxy_cpu);
    Array Pxy_real = slice(Pxy_real_full, 0, 0, static_cast<int>(freq_len));
    Pxx_sum = add(Pxx_sum, Pxy_real.to(Pxx_sum.dtype()));
  }

  // Average and scale using composite ops
  double scale = compute_psd_scale(nfft, fs, win, scaling, return_onesided);
  double avg_scale = scale / n_segments;
  Array Pxx = mul(Pxx_sum, full({freq_len}, avg_scale, DType::F64, cpu));

  // For one-sided, double the non-DC and non-Nyquist components
  if (return_onesided && nfft > 1) {
    Array mask = ones({freq_len}, DType::F64, cpu);
    // Set first and last to 1.0, middle to 2.0
    // Use put: mask[1:-1] = 2.0
    Array idx =
        arange(1.0, static_cast<double>(freq_len - 1), 1.0, DType::I64, cpu);
    Array twos = full({freq_len - 2}, 2.0, DType::F64, cpu);
    mask = put(mask, idx, twos);
    Pxx = mul(Pxx, mask);
  }

  Array f = fft_freqs(nfft, fs, return_onesided);

  return {f, Pxx};
}

// ============================================================================
// welch — Power Spectral Density
// ============================================================================

SpectralResult welch(const Array &x, double fs, const std::string &window,
                     int64_t nperseg, int64_t noverlap, int64_t nfft,
                     const std::string &detrend, bool return_onesided,
                     const std::string &scaling) {
  return csd(x, x, fs, window, nperseg, noverlap, nfft, detrend,
             return_onesided, scaling);
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

  // Cxy = |Pxy|^2 / (Pxx * Pyy) using composite ops
  Array pxy_sq = square(Pxy.Pxx);
  Array pxx_pyy = mul(Pxx.Pxx, Pyy.Pxx);

  // Avoid division by zero
  Array eps = full(pxx_pyy.shape(), 1e-30, pxx_pyy.dtype(), pxx_pyy.place());
  Array denominator = maximum(pxx_pyy, eps);
  Array cxy = div(pxy_sq, denominator);

  // Where denominator is too small, set to 0
  Array too_small = less_equal(pxx_pyy, eps);
  cxy = where(too_small, zeros_like(cxy), cxy);

  return {Pxx.f, cxy};
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
    noverlap = nperseg / 8;

  int64_t n = x_cpu.numel();
  int64_t step = nperseg - noverlap;
  int64_t n_segments = (n >= nperseg) ? ((n - nperseg) / step + 1) : 0;
  INS_CHECK(n_segments >= 1, "spectrogram: signal too short");

  int64_t freq_len = onesided_fft_len(nfft, return_onesided);

  Array win = get_window_for_spectral(window, nperseg);

  // Compute time array
  std::vector<double> t_arr(n_segments);
  for (int64_t i = 0; i < n_segments; ++i) {
    t_arr[i] = (i * step + nperseg / 2.0) / fs;
  }

  // Compute spectrogram segment by segment, accumulate into 2D Array
  Array Sxx_2d = zeros({freq_len, n_segments}, DType::F64, cpu);

  for (int64_t seg = 0; seg < n_segments; ++seg) {
    int64_t start = seg * step;

    // Extract and window segment
    Array seg_arr = slice(x_cpu, {0}, {start}, {start + nperseg});
    if (win.dtype() != seg_arr.dtype())
      win = win.to(seg_arr.dtype());
    Array windowed = mul(seg_arr, win);

    // Detrend
    if (detrend == "constant") {
      windowed = sub(windowed, mean(windowed));
    }

    // FFT
    Array seg_cpx = to_complex(windowed);
    Array Xf = fft::fft(seg_cpx, nfft);
    Array Xf_cpu = (Xf.place().kind() == DeviceKind::CPU) ? Xf : Xf.to(cpu);

    // Extract values based on mode using composite ops
    Array Xf_real;
    if (mode == "psd") {
      // |Xf|^2 = real^2 + imag^2
      Array r = real(Xf_cpu);
      Array im = imag(Xf_cpu);
      Xf_real = add(square(r), square(im));
    } else if (mode == "complex") {
      Xf_real = real(Xf_cpu);
    } else {
      Xf_real = abs(Xf_cpu);
    }

    // Store column using put or direct accumulation
    Array col = slice(Xf_real, 0, 0, static_cast<int>(freq_len));
    // Accumulate into Sxx_2d column by column
    // Build index for this column's elements
    Array row_idx =
        arange(0.0, static_cast<double>(freq_len), 1.0, DType::I64, cpu);
    Array col_idx =
        full({freq_len}, static_cast<int64_t>(seg), DType::I64, cpu);
    // Use scatter to write column
    Sxx_2d = scatter(Sxx_2d, 1, col_idx, reshape(col, {freq_len, 1}));
  }

  // Scale PSD using composite ops
  if (mode == "psd") {
    double scale = compute_psd_scale(nfft, fs, win, "density", return_onesided);
    Sxx_2d = mul(Sxx_2d, full({1}, scale, DType::F64, cpu));
    if (return_onesided && nfft > 1) {
      Array mask = ones({freq_len}, DType::F64, cpu);
      Array idx_mid =
          arange(1.0, static_cast<double>(freq_len - 1), 1.0, DType::I64, cpu);
      Array twos = full({freq_len - 2}, 2.0, DType::F64, cpu);
      mask = put(mask, idx_mid, twos);
      // Broadcast mask across segments: mask is [freq_len], Sxx_2d is
      // [freq_len, n_segments]
      Sxx_2d = mul(Sxx_2d, unsqueeze(mask, 1));
    }
  }

  Array f = fft_freqs(nfft, fs, return_onesided);
  Array t = to_array(t_arr, DType::F64, cpu);
  // Transpose from [freq_len, n_segments] to [n_segments, freq_len] for output
  Array S = transpose(Sxx_2d);

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

  // phasors = exp(2j*pi*events/period)
  Array t = mul(events, full(events.shape(), 2.0 * M_PI / period,
                             events.dtype(), events.place()));
  Array cos_val = cos(t);
  Array sin_val = sin(t);

  double real_sum = sum(cos_val).item<double>();
  double imag_sum = sum(sin_val).item<double>();

  int64_t n = events.numel();
  real_sum /= n;
  imag_sum /= n;

  double strength = std::sqrt(real_sum * real_sum + imag_sum * imag_sum);
  double angle = std::atan2(imag_sum, real_sum);

  return {strength, angle};
}

} // namespace signal
} // namespace ins
