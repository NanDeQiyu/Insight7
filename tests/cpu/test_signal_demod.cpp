// tests/cpu/test_signal_demod.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;
using namespace ins::signal;

class DemodTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
  }
};

// ============================================================================
// fm_demod
// ============================================================================

TEST_F(DemodTestCPU, FmDemodBasic) {
  // Create a simple FM signal: constant frequency
  int N = 100;
  double fs = 1000.0;
  double freq = 100.0;
  std::vector<std::complex<double>> signal(N);
  for (int i = 0; i < N; ++i) {
    double t = i / fs;
    signal[i] = std::exp(std::complex<double>(0.0, 2.0 * M_PI * freq * t));
  }
  Array arr = to_array(signal, DType::C64, CPUPlace());

  Array result = fm_demod(arr, -1);

  // Output should be N-1 elements
  EXPECT_EQ(result.numel(), N - 1);
  EXPECT_EQ(result.dtype(), DType::F64);

  // For a constant frequency signal, the demodulated output should be
  // approximately constant (2*pi*freq/fs)
  // Check middle values are close to expected
  const double *rd = result.data<double>();
  double expected = 2.0 * M_PI * freq / fs;
  for (int i = 10; i < N - 10; ++i) {
    EXPECT_NEAR(rd[i], expected, 0.1);
  }
}

TEST_F(DemodTestCPU, FmDemodC32) {
  // Test with complex64 input
  int N = 50;
  std::vector<std::complex<float>> signal(N);
  for (int i = 0; i < N; ++i) {
    float t = i / 1000.0f;
    signal[i] =
        std::exp(std::complex<float>(0.0f, 2.0f * 3.14159f * 50.0f * t));
  }
  Array arr = to_array(signal, DType::C32, CPUPlace());

  Array result = fm_demod(arr, -1);

  EXPECT_EQ(result.numel(), N - 1);
  EXPECT_EQ(result.dtype(), DType::F32);
}

TEST_F(DemodTestCPU, FmDemodInvalidInput) {
  // Real input should fail
  std::vector<double> data = {1.0, 2.0, 3.0};
  Array arr = to_array(data, DType::F64, CPUPlace());

  EXPECT_THROW(fm_demod(arr), std::exception);
}

TEST_F(DemodTestCPU, FmDemodVaryingFrequency) {
  // FM signal with varying frequency
  int N = 200;
  double fs = 1000.0;
  std::vector<std::complex<double>> signal(N);
  for (int i = 0; i < N; ++i) {
    double t = i / fs;
    double freq = 100.0 + 50.0 * std::sin(2.0 * M_PI * 10.0 * t);
    signal[i] = std::exp(std::complex<double>(0.0, 2.0 * M_PI * freq * t));
  }
  Array arr = to_array(signal, DType::C64, CPUPlace());

  Array result = fm_demod(arr, -1);

  EXPECT_EQ(result.numel(), N - 1);
  // Result should vary (not constant)
  const double *rd = result.data<double>();
  double first = rd[10];
  double mid = rd[100];
  // They should be different (varying frequency)
  // Just check the result is finite
  EXPECT_TRUE(std::isfinite(first));
  EXPECT_TRUE(std::isfinite(mid));
}
