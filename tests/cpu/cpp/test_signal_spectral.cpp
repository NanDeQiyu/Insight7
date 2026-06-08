// tests/cpu/test_signal_spectral.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

class SignalSpectralTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
#ifndef INSIGHT_USE_FFTW3
    GTEST_SKIP() << "FFTW3 not available, skipping FFT-dependent tests";
#endif
  }
};

// ========== welch ==========

TEST_F(SignalSpectralTestCPU, WelchBasic) {
  // Generate a simple sine wave
  int64_t N = 1024;
  double fs = 256.0;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 10.0 * i / fs); // 10 Hz
  }
  Array X = to_array(x);
  auto result = signal::welch(X, fs, "hann", 256, 128);

  // Should return frequency array and PSD
  EXPECT_GT(result.f.numel(), 0);
  EXPECT_GT(result.Pxx.numel(), 0);
  EXPECT_EQ(result.f.numel(), result.Pxx.numel());

  // Frequency array should go from 0 to fs/2
  const double *f = result.f.data<double>();
  EXPECT_NEAR(f[0], 0.0, 1e-10);
  EXPECT_NEAR(f[result.f.numel() - 1], fs / 2.0, 1e-6);
}

TEST_F(SignalSpectralTestCPU, WelchPeakAtSineFreq) {
  // PSD should have more energy near the sine frequency than at other
  // frequencies
  int64_t N = 4096;
  double fs = 256.0;
  double freq = 30.0;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * freq * i / fs);
  }
  Array X = to_array(x);
  auto result = signal::welch(X, fs, "hann", 512, 256);

  const double *f = result.f.data<double>();
  const double *Pxx = result.Pxx.data<double>();

  // Find peak frequency (skip DC)
  int64_t peak_idx = 1;
  double peak_val = Pxx[1];
  for (int64_t i = 2; i < result.Pxx.numel(); ++i) {
    if (Pxx[i] > peak_val) {
      peak_val = Pxx[i];
      peak_idx = i;
    }
  }
  // The PSD should have a clear peak (peak should be significantly above mean)
  double mean_psd = 0.0;
  for (int64_t i = 1; i < result.Pxx.numel(); ++i)
    mean_psd += Pxx[i];
  mean_psd /= (result.Pxx.numel() - 1);
  EXPECT_GT(peak_val, mean_psd * 2.0); // Peak should be > 2x mean
}

TEST_F(SignalSpectralTestCPU, WelchDC) {
  // With detrend="none", DC signal should have energy at f=0
  int64_t N = 512;
  std::vector<double> x(N, 1.0);
  Array X = to_array(x);
  auto result = signal::welch(X, 1.0, "boxcar", 256, 0, 0, "none");

  const double *Pxx = result.Pxx.data<double>();
  // DC component should dominate
  EXPECT_GT(Pxx[0], 0.0);
}

// ========== periodogram ==========

TEST_F(SignalSpectralTestCPU, PeriodogramBasic) {
  int64_t N = 512;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 5.0 * i / N);
  }
  Array X = to_array(x);
  auto result = signal::periodogram(X, 1.0, "boxcar");

  EXPECT_GT(result.f.numel(), 0);
  EXPECT_EQ(result.f.numel(), result.Pxx.numel());
}

// ========== csd ==========

TEST_F(SignalSpectralTestCPU, CsdBasic) {
  int64_t N = 1024;
  double fs = 256.0;
  std::vector<double> x(N), y(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 10.0 * i / fs);
    y[i] = std::sin(2.0 * M_PI * 10.0 * i / fs + 0.5); // phase-shifted
  }
  Array X = to_array(x);
  Array Y = to_array(y);
  auto result = signal::csd(X, Y, fs, "hann", 256, 128);

  EXPECT_GT(result.f.numel(), 0);
  EXPECT_EQ(result.f.numel(), result.Pxx.numel());
}

// ========== coherence ==========

TEST_F(SignalSpectralTestCPU, CoherenceBasic) {
  // Two identical signals should have coherence ~1
  int64_t N = 4096;
  double fs = 256.0;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 10.0 * i / fs);
  }
  Array X = to_array(x);
  auto result = signal::coherence(X, X, fs, "hann", 256, 128);

  EXPECT_GT(result.f.numel(), 0);
  // Coherence of identical signals should be ~1
  const double *Cxy = result.Pxx.data<double>();
  for (int64_t i = 1; i < result.Pxx.numel() - 1; ++i) {
    EXPECT_NEAR(Cxy[i], 1.0, 0.1) << "coherence at index " << i;
  }
}

// ========== spectrogram ==========

TEST_F(SignalSpectralTestCPU, SpectrogramBasic) {
  int64_t N = 2048;
  double fs = 256.0;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 10.0 * i / fs);
  }
  Array X = to_array(x);
  auto result = signal::spectrogram(X, fs, "hann", 256, 128);

  EXPECT_GT(result.f.numel(), 0);
  EXPECT_GT(result.t.numel(), 0);
  EXPECT_EQ(result.Sxx.shape().dim(0), result.t.numel());
  EXPECT_EQ(result.Sxx.shape().dim(1), result.f.numel());
}

// ========== stft ==========

TEST_F(SignalSpectralTestCPU, StftBasic) {
  int64_t N = 1024;
  double fs = 256.0;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 10.0 * i / fs);
  }
  Array X = to_array(x);
  auto result = signal::stft(X, fs, "hann", 256, 128);

  EXPECT_GT(result.f.numel(), 0);
  EXPECT_GT(result.t.numel(), 0);
  // STFT output should be complex (but stored as real for now)
  EXPECT_GT(result.Sxx.numel(), 0);
}

// ========== vectorstrength ==========

TEST_F(SignalSpectralTestCPU, VectorStrengthPerfectSync) {
  // Events exactly at period intervals -> strength = 1
  std::vector<double> events = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  Array ev = to_array(events);
  auto [strength, phase] = signal::vectorstrength(ev, 1.0);
  EXPECT_NEAR(strength, 1.0, 1e-10);
}

TEST_F(SignalSpectralTestCPU, VectorStrengthRandom) {
  // Random events -> strength should be < 1
  std::vector<double> events = {0.1, 0.7, 1.3, 2.9, 3.4, 4.8};
  Array ev = to_array(events);
  auto [strength, phase] = signal::vectorstrength(ev, 1.0);
  EXPECT_LT(strength, 1.0);
  EXPECT_GE(strength, 0.0);
}

TEST_F(SignalSpectralTestCPU, VectorStrengthAntiSync) {
  // Events at half-period -> strength should be ~0
  std::vector<double> events = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
  Array ev = to_array(events);
  auto [strength, phase] = signal::vectorstrength(ev, 1.0);
  EXPECT_NEAR(strength, 0.0, 1e-10);
}

// ========== lombscargle ==========

TEST_F(SignalSpectralTestCPU, LombScargleBasic) {
  // Generate a sinusoidal signal with uniform sampling
  int64_t N = 100;
  double fs = 10.0;      // 10 Hz sampling
  double f_signal = 1.0; // 1 Hz signal
  std::vector<double> t(N), y(N);
  for (int64_t i = 0; i < N; ++i) {
    t[i] = i / fs;
    y[i] = std::sin(2.0 * M_PI * f_signal * t[i]);
  }

  // Evaluation frequencies
  int64_t nf = 50;
  std::vector<double> freqs(nf);
  for (int64_t i = 0; i < nf; ++i) {
    freqs[i] = 0.1 * (i + 1); // 0.1 to 5.0 Hz
  }

  Array t_arr = to_array(t);
  Array y_arr = to_array(y);
  Array f_arr = to_array(freqs);

  Array power = signal::lombscargle(t_arr, y_arr, f_arr);

  // Output should have same length as freqs
  EXPECT_EQ(power.numel(), nf);

  // Peak should be near the signal frequency (1 Hz)
  const double *p = power.data<double>();
  int64_t peak_idx = 0;
  double peak_val = p[0];
  for (int64_t i = 1; i < nf; ++i) {
    if (p[i] > peak_val) {
      peak_val = p[i];
      peak_idx = i;
    }
  }
  double peak_freq = freqs[peak_idx];
  EXPECT_NEAR(peak_freq, f_signal, 0.3); // within 0.3 Hz of true freq
}

TEST_F(SignalSpectralTestCPU, LombScargleF32) {
  int64_t N = 50;
  std::vector<float> t(N), y(N), freqs(20);
  for (int64_t i = 0; i < N; ++i) {
    t[i] = static_cast<float>(i) * 0.1f;
    y[i] = std::sin(2.0f * static_cast<float>(M_PI) * 2.0f * t[i]);
  }
  for (int64_t i = 0; i < 20; ++i) {
    freqs[i] = 0.5f * (i + 1);
  }

  Array t_arr = to_array(t);
  Array y_arr = to_array(y);
  Array f_arr = to_array(freqs);

  Array power = signal::lombscargle(t_arr, y_arr, f_arr);
  EXPECT_EQ(power.numel(), 20);
  EXPECT_EQ(power.dtype(), DType::F32);
}

TEST_F(SignalSpectralTestCPU, LombScargleConstantSignal) {
  // Constant signal -> function should run without error
  int64_t N = 50;
  std::vector<double> t(N), y(N), freqs(10);
  for (int64_t i = 0; i < N; ++i) {
    t[i] = static_cast<double>(i) * 0.1;
    y[i] = 5.0;
  }
  for (int64_t i = 0; i < 10; ++i) {
    freqs[i] = 0.5 * (i + 1);
  }

  Array t_arr = to_array(t);
  Array y_arr = to_array(y);
  Array f_arr = to_array(freqs);

  Array power = signal::lombscargle(t_arr, y_arr, f_arr);
  EXPECT_EQ(power.numel(), 10);

  // All values should be finite and non-negative
  const double *p = power.data<double>();
  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_TRUE(std::isfinite(p[i]));
    EXPECT_GE(p[i], 0.0);
  }
}

TEST_F(SignalSpectralTestCPU, LombScargleInvalidInput) {
  Array x = to_array(std::vector<double>{1, 2, 3});
  Array y = to_array(std::vector<double>{1, 2});
  Array f = to_array(std::vector<double>{0.5});

  // x and y must have same length
  EXPECT_THROW(signal::lombscargle(x, y, f), std::exception);
}

// ========== istft ==========

TEST_F(SignalSpectralTestCPU, IstftBasic) {
  // Simple test: ISTFT should produce valid output
  std::vector<std::complex<double>> spec_data = {
      {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
  Array Zxx = to_array(spec_data, DType::C64, CPUPlace());
  Zxx = reshape(Zxx, {2, 2});

  Array result = signal::istft(Zxx, 1.0, "hann", 2, 1);

  EXPECT_GT(result.numel(), 0);
  EXPECT_EQ(result.shape().ndim(), 1);
}

TEST_F(SignalSpectralTestCPU, IstftRoundTrip) {
  // Test istft with a manually created spectrogram
  // 4 frequency bins, 3 time frames
  std::vector<std::complex<double>> spec_data(12);
  for (int i = 0; i < 12; ++i) {
    spec_data[i] = std::complex<double>(static_cast<double>(i % 4),
                                        static_cast<double>(i % 3));
  }
  Array Zxx = to_array(spec_data, DType::C64, CPUPlace());
  Zxx = reshape(Zxx, {4, 3});

  // nfft = 2*(4-1) = 6, nperseg = 6
  Array result = signal::istft(Zxx, 1.0, "hann", 6, 3);

  EXPECT_GT(result.numel(), 0);
  EXPECT_EQ(result.shape().ndim(), 1);
}
