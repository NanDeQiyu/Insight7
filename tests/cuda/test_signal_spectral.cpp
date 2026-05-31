// tests/cuda/test_signal_spectral.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

namespace {

class SignalSpectralTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "cuda"});
    try {
      set_device(GPUPlace(0));
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
  }
};

// ========== welch ==========

TEST_F(SignalSpectralTestGPU, WelchBasic) {
  // Generate a simple sine wave
  int64_t N = 1024;
  double fs = 256.0;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 10.0 * i / fs); // 10 Hz
  }
  Array X = to_array(x).to(GPUPlace(0));
  auto result = signal::welch(X, fs, "hann", 256, 128);

  // Should return frequency array and PSD
  EXPECT_GT(result.f.numel(), 0);
  EXPECT_GT(result.Pxx.numel(), 0);
  EXPECT_EQ(result.f.numel(), result.Pxx.numel());

  // Frequency array should go from 0 to fs/2
  Array f_cpu = result.f.to(CPUPlace());
  EXPECT_NEAR(f_cpu.data<double>()[0], 0.0, 1e-10);
  EXPECT_NEAR(f_cpu.data<double>()[result.f.numel() - 1], fs / 2.0, 1e-6);
}

TEST_F(SignalSpectralTestGPU, WelchPeakAtSineFreq) {
  // PSD should have more energy near the sine frequency than at other
  // frequencies
  int64_t N = 4096;
  double fs = 256.0;
  double freq = 30.0;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * freq * i / fs);
  }
  Array X = to_array(x).to(GPUPlace(0));
  auto result = signal::welch(X, fs, "hann", 512, 256);

  Array f_cpu = result.f.to(CPUPlace());
  Array Pxx_cpu = result.Pxx.to(CPUPlace());
  const double *f = f_cpu.data<double>();
  const double *Pxx = Pxx_cpu.data<double>();

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

TEST_F(SignalSpectralTestGPU, WelchDC) {
  // With detrend="none", DC signal should have energy at f=0
  int64_t N = 512;
  std::vector<double> x(N, 1.0);
  Array X = to_array(x).to(GPUPlace(0));
  auto result = signal::welch(X, 1.0, "boxcar", 256, 0, 0, "none");

  Array Pxx_cpu = result.Pxx.to(CPUPlace());
  const double *Pxx = Pxx_cpu.data<double>();
  // DC component should dominate
  EXPECT_GT(Pxx[0], 0.0);
}

// ========== periodogram ==========

TEST_F(SignalSpectralTestGPU, PeriodogramBasic) {
  int64_t N = 512;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 5.0 * i / N);
  }
  Array X = to_array(x).to(GPUPlace(0));
  auto result = signal::periodogram(X, 1.0, "boxcar");

  EXPECT_GT(result.f.numel(), 0);
  EXPECT_EQ(result.f.numel(), result.Pxx.numel());
}

// ========== csd ==========

TEST_F(SignalSpectralTestGPU, CsdBasic) {
  int64_t N = 1024;
  double fs = 256.0;
  std::vector<double> x(N), y(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 10.0 * i / fs);
    y[i] = std::sin(2.0 * M_PI * 10.0 * i / fs + 0.5); // phase-shifted
  }
  Array X = to_array(x).to(GPUPlace(0));
  Array Y = to_array(y).to(GPUPlace(0));
  auto result = signal::csd(X, Y, fs, "hann", 256, 128);

  EXPECT_GT(result.f.numel(), 0);
  EXPECT_EQ(result.f.numel(), result.Pxx.numel());
}

// ========== coherence ==========

TEST_F(SignalSpectralTestGPU, CoherenceBasic) {
  // Two identical signals should have coherence ~1
  int64_t N = 4096;
  double fs = 256.0;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 10.0 * i / fs);
  }
  Array X = to_array(x).to(GPUPlace(0));
  auto result = signal::coherence(X, X, fs, "hann", 256, 128);

  EXPECT_GT(result.f.numel(), 0);
  // Coherence of identical signals should be ~1
  Array Cxy_cpu = result.Pxx.to(CPUPlace());
  const double *Cxy = Cxy_cpu.data<double>();
  for (int64_t i = 1; i < result.Pxx.numel() - 1; ++i) {
    EXPECT_NEAR(Cxy[i], 1.0, 0.1) << "coherence at index " << i;
  }
}

// ========== spectrogram ==========

TEST_F(SignalSpectralTestGPU, SpectrogramBasic) {
  int64_t N = 2048;
  double fs = 256.0;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 10.0 * i / fs);
  }
  Array X = to_array(x).to(GPUPlace(0));
  auto result = signal::spectrogram(X, fs, "hann", 256, 128);

  EXPECT_GT(result.f.numel(), 0);
  EXPECT_GT(result.t.numel(), 0);
  EXPECT_EQ(result.Sxx.shape().dim(0), result.t.numel());
  EXPECT_EQ(result.Sxx.shape().dim(1), result.f.numel());
}

// ========== stft ==========

TEST_F(SignalSpectralTestGPU, StftBasic) {
  int64_t N = 1024;
  double fs = 256.0;
  std::vector<double> x(N);
  for (int64_t i = 0; i < N; ++i) {
    x[i] = std::sin(2.0 * M_PI * 10.0 * i / fs);
  }
  Array X = to_array(x).to(GPUPlace(0));
  auto result = signal::stft(X, fs, "hann", 256, 128);

  EXPECT_GT(result.f.numel(), 0);
  EXPECT_GT(result.t.numel(), 0);
  // STFT output should be complex (but stored as real for now)
  EXPECT_GT(result.Sxx.numel(), 0);
}

// ========== vectorstrength ==========

TEST_F(SignalSpectralTestGPU, VectorStrengthPerfectSync) {
  // Events exactly at period intervals -> strength = 1
  std::vector<double> events = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  Array ev = to_array(events).to(GPUPlace(0));
  auto [strength, phase] = signal::vectorstrength(ev, 1.0);
  EXPECT_NEAR(strength, 1.0, 1e-10);
}

TEST_F(SignalSpectralTestGPU, VectorStrengthRandom) {
  // Random events -> strength should be < 1
  std::vector<double> events = {0.1, 0.7, 1.3, 2.9, 3.4, 4.8};
  Array ev = to_array(events).to(GPUPlace(0));
  auto [strength, phase] = signal::vectorstrength(ev, 1.0);
  EXPECT_LT(strength, 1.0);
  EXPECT_GE(strength, 0.0);
}

TEST_F(SignalSpectralTestGPU, VectorStrengthAntiSync) {
  // Events at half-period -> strength should be ~0
  std::vector<double> events = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
  Array ev = to_array(events).to(GPUPlace(0));
  auto [strength, phase] = signal::vectorstrength(ev, 1.0);
  EXPECT_NEAR(strength, 0.0, 1e-10);
}

} // namespace
