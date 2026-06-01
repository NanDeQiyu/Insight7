// tests/cpu/test_signal_wavelets.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

class SignalWaveletsTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
  }
};

// ========== morlet ==========

TEST_F(SignalWaveletsTestCPU, MorletBasic) {
  Array w = signal::morlet(100, 5.0, 1.0, true);
  EXPECT_EQ(w.numel(), 100);
  EXPECT_EQ(w.dtype(), DType::C64);
}

TEST_F(SignalWaveletsTestCPU, MorletEnergy) {
  // Morlet wavelet should have non-zero energy
  Array w = signal::morlet(200, 5.0, 1.0, true);
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(w.data<char>());
  double energy = 0.0;
  for (int64_t i = 0; i < 200; ++i) {
    energy += std::norm(data[i]);
  }
  EXPECT_GT(energy, 0.0);
}

TEST_F(SignalWaveletsTestCPU, MorletCentered) {
  // Peak should be near the center
  Array w = signal::morlet(200, 5.0, 1.0, true);
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(w.data<char>());
  int64_t peak_idx = 0;
  double peak_mag = 0;
  for (int64_t i = 0; i < 200; ++i) {
    double mag = std::abs(data[i]);
    if (mag > peak_mag) {
      peak_mag = mag;
      peak_idx = i;
    }
  }
  // Peak should be near center (index ~100)
  EXPECT_NEAR(peak_idx, 100, 20);
}

TEST_F(SignalWaveletsTestCPU, MorletComplete) {
  // Complete vs standard should differ
  Array w_std = signal::morlet(100, 5.0, 1.0, false);
  Array w_full = signal::morlet(100, 5.0, 1.0, true);
  // They should be different (the correction term)
  const std::complex<double> *d1 =
      reinterpret_cast<const std::complex<double> *>(w_std.data<char>());
  const std::complex<double> *d2 =
      reinterpret_cast<const std::complex<double> *>(w_full.data<char>());
  bool differs = false;
  for (int64_t i = 0; i < 100; ++i) {
    if (std::abs(d1[i] - d2[i]) > 1e-10) {
      differs = true;
      break;
    }
  }
  EXPECT_TRUE(differs);
}

// ========== ricker ==========

TEST_F(SignalWaveletsTestCPU, RickerBasic) {
  Array w = signal::ricker(100, 5.0);
  EXPECT_EQ(w.numel(), 100);
  EXPECT_EQ(w.dtype(), DType::F64);
}

TEST_F(SignalWaveletsTestCPU, RickerSymmetric) {
  // Ricker wavelet should be symmetric around center
  Array w = signal::ricker(101, 5.0);
  const double *data = w.data<double>();
  for (int64_t i = 0; i < 50; ++i) {
    EXPECT_NEAR(data[i], data[100 - i], 1e-10) << "symmetry failed at " << i;
  }
}

TEST_F(SignalWaveletsTestCPU, RickerPeakAtCenter) {
  // Peak should be at center
  Array w = signal::ricker(101, 5.0);
  const double *data = w.data<double>();
  int64_t peak_idx = 0;
  for (int64_t i = 1; i < 101; ++i) {
    if (data[i] > data[peak_idx])
      peak_idx = i;
  }
  EXPECT_EQ(peak_idx, 50);
}

TEST_F(SignalWaveletsTestCPU, RickerZeroCrossing) {
  // Ricker should have zero crossings
  Array w = signal::ricker(101, 5.0);
  const double *data = w.data<double>();
  // Center is positive, edges should be negative
  EXPECT_GT(data[50], 0.0);
  EXPECT_LT(data[0], 0.0);
  EXPECT_LT(data[100], 0.0);
}

TEST_F(SignalWaveletsTestCPU, RickerDifferentWidths) {
  Array w1 = signal::ricker(101, 3.0);
  Array w2 = signal::ricker(101, 10.0);
  // Wider wavelet should have broader peak
  const double *d1 = w1.data<double>();
  const double *d2 = w2.data<double>();
  // The narrower wavelet should have a higher peak
  EXPECT_GT(d1[50], d2[50]);
}

// ========== morlet2 ==========

TEST_F(SignalWaveletsTestCPU, Morlet2Basic) {
  Array w = signal::morlet2(100, 5.0, 5.0);
  EXPECT_EQ(w.numel(), 100);
  EXPECT_EQ(w.dtype(), DType::C64);
}

TEST_F(SignalWaveletsTestCPU, Morlet2Energy) {
  Array w = signal::morlet2(200, 10.0, 5.0);
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(w.data<char>());
  double energy = 0.0;
  for (int64_t i = 0; i < 200; ++i) {
    energy += std::norm(data[i]);
  }
  EXPECT_GT(energy, 0.0);
}

// ========== cwt ==========

#ifdef INSIGHT_USE_FFTW3

TEST_F(SignalWaveletsTestCPU, CwtBasic) {
  // Simple signal
  int64_t N = 256;
  std::vector<double> data(N);
  for (int64_t i = 0; i < N; ++i) {
    data[i] = std::sin(2.0 * M_PI * 10.0 * i / N);
  }
  Array X = to_array(data);

  std::vector<double> widths = {1.0, 2.0, 4.0, 8.0};
  Array result = signal::cwt(X, signal::ricker, widths);

  EXPECT_EQ(result.shape().dim(0), 4);
  EXPECT_EQ(result.shape().dim(1), N);
}

TEST_F(SignalWaveletsTestCPU, CwtRickerPeak) {
  // CWT with ricker should detect the sine wave
  int64_t N = 512;
  std::vector<double> data(N);
  for (int64_t i = 0; i < N; ++i) {
    data[i] = std::sin(2.0 * M_PI * 10.0 * i / N);
  }
  Array X = to_array(data);

  std::vector<double> widths = {5.0};
  Array result = signal::cwt(X, signal::ricker, widths);
  EXPECT_EQ(result.shape().dim(0), 1);
  EXPECT_EQ(result.shape().dim(1), N);
}

#endif // INSIGHT_USE_FFTW3
