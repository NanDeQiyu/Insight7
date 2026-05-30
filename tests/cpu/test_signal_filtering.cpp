// tests/cpu/test_signal_filtering.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

class SignalFilteringTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
#ifndef INSIGHT_USE_FFTW3
    GTEST_SKIP() << "FFTW3 not available, skipping FFT-dependent tests";
#endif
  }
};

namespace {
template <typename T>
void expect_float_equal(const Array &arr, const std::vector<T> &expected,
                        T tol = 1e-6) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_NEAR(data[i], expected[i], tol) << "mismatch at index " << i;
  }
}
} // namespace

// ========== hilbert ==========

TEST_F(SignalFilteringTestCPU, HilbertBasic) {
  // Hilbert transform produces a complex analytic signal
  int64_t N = 64;
  std::vector<double> x_data(N);
  for (int64_t i = 0; i < N; ++i) {
    x_data[i] = std::cos(2.0 * M_PI * i / N);
  }
  Array x = to_array(x_data);
  Array y = signal::hilbert(x, N);

  ASSERT_EQ(y.dtype(), DType::C64);
  ASSERT_EQ(y.numel(), N);

  // The analytic signal should have non-zero magnitude everywhere
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y.data<char>());
  for (int64_t i = 0; i < N; ++i) {
    EXPECT_GT(std::abs(data[i]), 0.01) << "near-zero magnitude at " << i;
  }
}

TEST_F(SignalFilteringTestCPU, HilbertDC) {
  // Hilbert transform of DC should be zero (imaginary part)
  std::vector<double> x_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  Array x = to_array(x_data);
  Array y = signal::hilbert(x, 8);
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y.data<char>());
  // Real part should be 1.0 (DC preserved)
  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(data[i].real(), 1.0, 1e-10);
    EXPECT_NEAR(data[i].imag(), 0.0, 1e-10);
  }
}

TEST_F(SignalFilteringTestCPU, HilbertDefaultN) {
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0};
  Array x = to_array(x_data);
  Array y = signal::hilbert(x); // N defaults to x.numel()
  EXPECT_EQ(y.numel(), 4);
  EXPECT_EQ(y.dtype(), DType::C64);
}

// ========== detrend ==========

TEST_F(SignalFilteringTestCPU, DetrendConstant) {
  // Remove constant (mean)
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array x = to_array(x_data);
  Array y = signal::detrend(x, -1, "constant");

  // Mean = 3.0, so detrended = [-2, -1, 0, 1, 2]
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], -2.0, 1e-10);
  EXPECT_NEAR(d[1], -1.0, 1e-10);
  EXPECT_NEAR(d[2], 0.0, 1e-10);
  EXPECT_NEAR(d[3], 1.0, 1e-10);
  EXPECT_NEAR(d[4], 2.0, 1e-10);
}

TEST_F(SignalFilteringTestCPU, DetrendLinear) {
  // Remove linear trend from [0, 1, 2, 3, 4] + trend = [0, 2, 4, 6, 8]
  // Input: [0, 3, 6, 9, 12] = linear trend with slope 3
  // After linear detrend, should be all zeros
  std::vector<double> x_data = {0.0, 3.0, 6.0, 9.0, 12.0};
  Array x = to_array(x_data);
  Array y = signal::detrend(x, -1, "linear");
  const double *d = y.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], 0.0, 1e-10) << "mismatch at " << i;
  }
}

TEST_F(SignalFilteringTestCPU, DetrendLinearWithNoise) {
  // Add small perturbation to linear signal
  std::vector<double> x_data = {0.1, 3.0, 5.9, 9.1, 12.0};
  Array x = to_array(x_data);
  Array y = signal::detrend(x, -1, "linear");
  // After detrend, residuals should be small
  const double *d = y.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], 0.0, 0.2) << "mismatch at " << i;
  }
}

// ========== lfilter (FIR case) ==========

TEST_F(SignalFilteringTestCPU, LfilterFIRBasic) {
  // b = [1, 1], a = [1], x = [1, 0, 0, 0, 0]
  // y = convolve([1,1], [1,0,0,0,0]) = [1, 1, 0, 0, 0, 0]
  std::vector<double> b_data = {1.0, 1.0};
  std::vector<double> a_data = {1.0};
  std::vector<double> x_data = {1.0, 0.0, 0.0, 0.0, 0.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x = to_array(x_data);
  Array y = signal::lfilter(b, a, x);
  EXPECT_EQ(y.numel(), 6);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 1.0, 1e-10);
  for (int i = 2; i < 6; ++i) {
    EXPECT_NEAR(d[i], 0.0, 1e-10);
  }
}

TEST_F(SignalFilteringTestCPU, LfilterFIRMovingAvg) {
  // 3-point moving average: b = [1/3, 1/3, 1/3], a = [1]
  std::vector<double> b_data = {1.0 / 3, 1.0 / 3, 1.0 / 3};
  std::vector<double> a_data = {1.0};
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x = to_array(x_data);
  Array y = signal::lfilter(b, a, x);
  // Full convolution: [1/3, 1, 2, 3, 4, 10/3, 5/3]
  EXPECT_NEAR(y.data<double>()[1], 1.0, 1e-10);
  EXPECT_NEAR(y.data<double>()[2], 2.0, 1e-10);
  EXPECT_NEAR(y.data<double>()[3], 3.0, 1e-10);
  EXPECT_NEAR(y.data<double>()[4], 4.0, 1e-10);
}

// ========== lfilter (IIR case) ==========

TEST_F(SignalFilteringTestCPU, LfilterIIRBasic) {
  // Simple first-order IIR: y[n] = x[n] + 0.5*y[n-1]
  // b = [1], a = [1, -0.5]
  std::vector<double> b_data = {1.0};
  std::vector<double> a_data = {1.0, -0.5};
  // x = [1, 0, 0, 0, 0] (impulse)
  std::vector<double> x_data = {1.0, 0.0, 0.0, 0.0, 0.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x = to_array(x_data);
  Array y = signal::lfilter(b, a, x);
  // y = [1, 0.5, 0.25, 0.125, 0.0625]
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 0.5, 1e-10);
  EXPECT_NEAR(d[2], 0.25, 1e-10);
  EXPECT_NEAR(d[3], 0.125, 1e-10);
  EXPECT_NEAR(d[4], 0.0625, 1e-10);
}

// ========== lfilter_zi ==========

TEST_F(SignalFilteringTestCPU, LfilterZiFIR) {
  // FIR: b=[1,1], a=[1] -> zi should be empty or trivial
  std::vector<double> b_data = {1.0, 1.0};
  std::vector<double> a_data = {1.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array zi = signal::lfilter_zi(b, a);
  // For FIR (na=1), zi length = max(nb,na)-1 = 1
  // zi = b[1]/a[0] = 1.0
  EXPECT_EQ(zi.numel(), 1);
  EXPECT_NEAR(zi.data<double>()[0], 1.0, 1e-10);
}

TEST_F(SignalFilteringTestCPU, LfilterZiIIR) {
  // b=[1], a=[1, -0.5] -> zi = b[1]-a[1]*b[0] / (1-a[1]) = 0.5/0.5 = 1.0
  std::vector<double> b_data = {1.0};
  std::vector<double> a_data = {1.0, -0.5};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array zi = signal::lfilter_zi(b, a);
  EXPECT_EQ(zi.numel(), 1);
  EXPECT_NEAR(zi.data<double>()[0], 1.0, 1e-10);
}

// ========== filtfilt ==========

TEST_F(SignalFilteringTestCPU, FiltfiltFIR) {
  // Zero-phase filtering with b=[1,1], a=[1]
  std::vector<double> b_data = {1.0, 1.0};
  std::vector<double> a_data = {1.0};
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x = to_array(x_data);
  Array y = signal::filtfilt(b, a, x);
  // filtfilt should have zero phase distortion
  // The output should be symmetric if input is symmetric
  EXPECT_EQ(y.numel(), x.numel());
}

// ========== freq_shift ==========

TEST_F(SignalFilteringTestCPU, FreqShiftBasic) {
  // Shift a DC signal by some frequency
  int64_t N = 64;
  std::vector<double> x_data(N, 1.0);
  Array x = to_array(x_data);
  Array y = signal::freq_shift(x, 0.0, 1.0);
  // freq_shift of 0 should return the original signal (as complex)
  ASSERT_EQ(y.dtype(), DType::C64);
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y.data<char>());
  for (int64_t i = 0; i < N; ++i) {
    EXPECT_NEAR(std::abs(data[i]), 1.0, 1e-10);
  }
}

TEST_F(SignalFilteringTestCPU, FreqShiftPositive) {
  // Shift a signal by fs/4 (quarter sample rate)
  int64_t N = 16;
  std::vector<double> x_data(N, 1.0);
  Array x = to_array(x_data);
  double fs = 1.0;
  Array y = signal::freq_shift(x, fs / 4.0, fs);
  // Should produce a complex sinusoid at fs/4
  ASSERT_EQ(y.dtype(), DType::C64);
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y.data<char>());
  // At freq=fs/4, phase increments by pi/2 per sample
  for (int64_t i = 0; i < N; ++i) {
    double expected_real = std::cos(2.0 * M_PI * 0.25 * i);
    double expected_imag = std::sin(2.0 * M_PI * 0.25 * i);
    EXPECT_NEAR(data[i].real(), expected_real, 1e-10);
    EXPECT_NEAR(data[i].imag(), expected_imag, 1e-10);
  }
}

// ========== decimate ==========

TEST_F(SignalFilteringTestCPU, DecimateBasic) {
  // Create a smooth signal and decimate by 2
  int64_t N = 100;
  std::vector<double> x_data(N);
  for (int64_t i = 0; i < N; ++i) {
    x_data[i] = std::sin(2.0 * M_PI * i / N);
  }
  Array x = to_array(x_data);
  Array y = signal::decimate(x, 2, -1, false);
  // Output length should be ceil(N/2) = 50
  EXPECT_EQ(y.numel(), 50);
}

TEST_F(SignalFilteringTestCPU, DecimateZeroPhase) {
  int64_t N = 100;
  std::vector<double> x_data(N);
  for (int64_t i = 0; i < N; ++i) {
    x_data[i] = std::sin(2.0 * M_PI * i / N);
  }
  Array x = to_array(x_data);
  Array y = signal::decimate(x, 4, -1, true);
  EXPECT_EQ(y.numel(), 25);
}

// ========== resample ==========

TEST_F(SignalFilteringTestCPU, ResampleIdentity) {
  // Resample to same length -> should return same signal
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array x = to_array(x_data);
  Array y = signal::resample(x, 5);
  EXPECT_EQ(y.numel(), 5);
  const double *d = y.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], x_data[i], 1e-6);
  }
}

TEST_F(SignalFilteringTestCPU, ResampleUpsample) {
  // Upsample a 4-point signal to 8 points
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0};
  Array x = to_array(x_data);
  Array y = signal::resample(x, 8);
  EXPECT_EQ(y.numel(), 8);
}

TEST_F(SignalFilteringTestCPU, ResampleDownsample) {
  // Downsample an 8-point signal to 4 points
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  Array x = to_array(x_data);
  Array y = signal::resample(x, 4);
  EXPECT_EQ(y.numel(), 4);
}

// ========== resample_poly ==========

TEST_F(SignalFilteringTestCPU, ResamplePolyIdentity) {
  // up=1, down=1 -> identity
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array x = to_array(x_data);
  Array y = signal::resample_poly(x, 1, 1);
  EXPECT_EQ(y.numel(), 5);
  const double *d = y.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], x_data[i], 1e-10);
  }
}

TEST_F(SignalFilteringTestCPU, ResamplePolyUpsample) {
  // up=2, down=1 -> double the length
  std::vector<double> x_data = {1.0, 0.0, 1.0, 0.0};
  Array x = to_array(x_data);
  Array y = signal::resample_poly(x, 2, 1);
  // Output length should be approximately 2*4 = 8
  EXPECT_GE(y.numel(), 7);
  EXPECT_LE(y.numel(), 9);
}

TEST_F(SignalFilteringTestCPU, ResamplePolyDownsample) {
  // up=1, down=2 -> halve the length
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  Array x = to_array(x_data);
  Array y = signal::resample_poly(x, 1, 2);
  // Output length should be approximately 8/2 = 4
  EXPECT_GE(y.numel(), 3);
  EXPECT_LE(y.numel(), 5);
}

// ========== firfilter ==========

TEST_F(SignalFilteringTestCPU, FirfilterBasic) {
  // b = [1, 2, 1], x = [1, 0, 0, 0, 0]
  // y = convolve([1,2,1], [1,0,0,0,0]) = [1, 2, 1, 0, 0, 0, 0]
  std::vector<double> b_data = {1.0, 2.0, 1.0};
  std::vector<double> x_data = {1.0, 0.0, 0.0, 0.0, 0.0};
  Array b = to_array(b_data);
  Array x = to_array(x_data);
  Array y = signal::firfilter(b, x);
  EXPECT_EQ(y.numel(), 7);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 2.0, 1e-10);
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  for (int i = 3; i < 7; ++i) {
    EXPECT_NEAR(d[i], 0.0, 1e-10);
  }
}
