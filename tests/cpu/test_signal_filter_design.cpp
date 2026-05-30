// tests/cpu/test_signal_filter_design.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

class SignalFilterDesignTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
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

// ========== kaiser_beta ==========

TEST_F(SignalFilterDesignTestCPU, KaiserBetaLow) {
  // a <= 21 -> beta = 0
  EXPECT_DOUBLE_EQ(signal::kaiser_beta(0.0), 0.0);
  EXPECT_DOUBLE_EQ(signal::kaiser_beta(21.0), 0.0);
}

TEST_F(SignalFilterDesignTestCPU, KaiserBetaMid) {
  // 21 < a <= 50: 0.5842*(a-21)^0.4 + 0.07886*(a-21)
  double a = 40.0;
  double expected = 0.5842 * std::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0);
  EXPECT_NEAR(signal::kaiser_beta(a), expected, 1e-10);
}

TEST_F(SignalFilterDesignTestCPU, KaiserBetaHigh) {
  // a > 50: 0.1102*(a-8.7)
  double a = 60.0;
  double expected = 0.1102 * (a - 8.7);
  EXPECT_NEAR(signal::kaiser_beta(a), expected, 1e-10);
}

TEST_F(SignalFilterDesignTestCPU, KaiserBetaBoundary) {
  // At a=50, both formulas should give similar results
  double a = 50.0;
  double beta = signal::kaiser_beta(a);
  EXPECT_GT(beta, 4.0); // Should be around 4.55
  EXPECT_LT(beta, 6.0);
}

// ========== kaiser_atten ==========

TEST_F(SignalFilterDesignTestCPU, KaiserAttenBasic) {
  // a = 2.285 * (numtaps-1) * pi * width + 7.95
  double atten = signal::kaiser_atten(101, 0.1);
  double expected = 2.285 * 100.0 * M_PI * 0.1 + 7.95;
  EXPECT_NEAR(atten, expected, 1e-10);
}

TEST_F(SignalFilterDesignTestCPU, KaiserAttenNarrow) {
  // Larger width -> higher attenuation (Kaiser formula)
  double atten_wide = signal::kaiser_atten(101, 0.2);
  double atten_narrow = signal::kaiser_atten(101, 0.05);
  EXPECT_GT(atten_wide, atten_narrow);
}

// ========== firwin ==========

TEST_F(SignalFilterDesignTestCPU, FirwinLowpassBasic) {
  // Lowpass FIR with 11 taps, cutoff at 0.3
  Array h = signal::firwin(11, {0.3}, "hamming", "lowpass", true);
  EXPECT_EQ(h.numel(), 11);
  // Symmetric (Type I: odd numtaps)
  const double *d = h.data<double>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], d[10 - i], 1e-10) << "symmetry failed at index " << i;
  }
}

TEST_F(SignalFilterDesignTestCPU, FirwinLowpassDCGain) {
  // Scaled lowpass: DC gain should be 1
  Array h = signal::firwin(21, {0.25}, "hamming", "lowpass", true);
  Array dc = sum(h);
  EXPECT_NEAR(dc.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalFilterDesignTestCPU, FirwinHighpassBasic) {
  // Highpass FIR with 11 taps, cutoff at 0.3
  Array h = signal::firwin(11, {0.3}, "hamming", "highpass", true);
  EXPECT_EQ(h.numel(), 11);
  // Symmetric
  const double *d = h.data<double>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], d[10 - i], 1e-10) << "symmetry failed at index " << i;
  }
}

TEST_F(SignalFilterDesignTestCPU, FirwinHighpassNyquistGain) {
  // Scaled highpass: Nyquist gain should be 1
  // H(Nyquist) = sum(h * (-1)^n)
  Array h = signal::firwin(21, {0.25}, "hamming", "highpass", true);
  const double *d = h.data<double>();
  double nyquist_gain = 0.0;
  for (int i = 0; i < 21; ++i) {
    nyquist_gain += d[i] * (i % 2 == 0 ? 1.0 : -1.0);
  }
  EXPECT_NEAR(nyquist_gain, 1.0, 1e-6);
}

TEST_F(SignalFilterDesignTestCPU, FirwinBandpass) {
  // Bandpass FIR with 21 taps
  Array h = signal::firwin(21, {0.2, 0.5}, "hamming", "bandpass", false);
  EXPECT_EQ(h.numel(), 21);
  // Sum (DC gain) should be near 0 for bandpass (windowed sinc approximation)
  Array dc = sum(h);
  EXPECT_NEAR(dc.data<double>()[0], 0.0, 0.01);
}

TEST_F(SignalFilterDesignTestCPU, FirwinBandstop) {
  // Bandstop FIR with 21 taps
  Array h = signal::firwin(21, {0.2, 0.5}, "hamming", "bandstop", false);
  EXPECT_EQ(h.numel(), 21);
}

TEST_F(SignalFilterDesignTestCPU, FirwinDifferentWindows) {
  // Test with different windows
  Array h1 = signal::firwin(11, {0.3}, "hann", "lowpass", false);
  Array h2 = signal::firwin(11, {0.3}, "blackman", "lowpass", false);
  Array h3 = signal::firwin(11, {0.3}, "kaiser", "lowpass", false);
  EXPECT_EQ(h1.numel(), 11);
  EXPECT_EQ(h2.numel(), 11);
  EXPECT_EQ(h3.numel(), 11);
  // Different windows should give different results
  EXPECT_NE(h1.data<double>()[0], h2.data<double>()[0]);
}

TEST_F(SignalFilterDesignTestCPU, FirwinUnscaled) {
  // Unscaled: DC gain should NOT be 1
  Array h = signal::firwin(21, {0.25}, "hamming", "lowpass", false);
  Array dc = sum(h);
  // The unscaled DC gain is the sum of the window (hamming sum / N ≈ 0.54)
  // It should be different from 1.0
  double dc_val = dc.data<double>()[0];
  EXPECT_NE(dc_val, 1.0);
}

// ========== firwin2 ==========

TEST_F(SignalFilterDesignTestCPU, Firwin2Lowpass) {
  // Lowpass: gain=1 at f=0..0.5, gain=0 at f=0.5..1
  std::vector<double> freq = {0.0, 0.5, 0.5, 1.0};
  std::vector<double> gain = {1.0, 1.0, 0.0, 0.0};
  Array h = signal::firwin2(15, freq, gain, 0, "hamming", false);
  EXPECT_EQ(h.numel(), 15);
  // DC gain should be close to 1
  Array dc = sum(h);
  EXPECT_NEAR(dc.data<double>()[0], 1.0, 0.1);
}

TEST_F(SignalFilterDesignTestCPU, Firwin2Highpass) {
  // Highpass: gain=0 at f=0..0.5, gain=1 at f=0.5..1
  std::vector<double> freq = {0.0, 0.5, 0.5, 1.0};
  std::vector<double> gain = {0.0, 0.0, 1.0, 1.0};
  Array h = signal::firwin2(15, freq, gain, 0, "hamming", false);
  EXPECT_EQ(h.numel(), 15);
}

TEST_F(SignalFilterDesignTestCPU, Firwin2Bandpass) {
  // Bandpass: gain=1 at f=0.2..0.5, 0 elsewhere
  std::vector<double> freq = {0.0, 0.2, 0.2, 0.5, 0.5, 1.0};
  std::vector<double> gain = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
  Array h = signal::firwin2(21, freq, gain, 0, "hamming", false);
  EXPECT_EQ(h.numel(), 21);
}

TEST_F(SignalFilterDesignTestCPU, Firwin2DifferentLengths) {
  std::vector<double> freq = {0.0, 0.5, 1.0};
  std::vector<double> gain = {1.0, 1.0, 0.0};
  Array h_short = signal::firwin2(7, freq, gain);
  Array h_long = signal::firwin2(31, freq, gain);
  EXPECT_EQ(h_short.numel(), 7);
  EXPECT_EQ(h_long.numel(), 31);
}

// ========== cmplx_sort ==========

TEST_F(SignalFilterDesignTestCPU, CmplxSortReal) {
  // Sort real values by absolute value
  std::vector<double> data = {3.0, -1.0, 2.0, -4.0};
  Array arr = to_array(data);
  Array sorted = signal::cmplx_sort(arr);
  const double *d = sorted.data<double>();
  EXPECT_NEAR(d[0], -1.0, 1e-10); // |−1| = 1
  EXPECT_NEAR(d[1], 2.0, 1e-10);  // |2| = 2
  EXPECT_NEAR(d[2], 3.0, 1e-10);  // |3| = 3
  EXPECT_NEAR(d[3], -4.0, 1e-10); // |−4| = 4
}

TEST_F(SignalFilterDesignTestCPU, CmplxSortComplex) {
  // Sort complex values by magnitude
  std::vector<std::complex<double>> data = {
      {0, 5}, // |5|
      {1, 0}, // |1|
      {0, 3}, // |3|
      {4, 0}, // |4|
  };
  Array arr = to_array(data, DType::C64);
  Array sorted = signal::cmplx_sort(arr);
  const std::complex<double> *d =
      reinterpret_cast<const std::complex<double> *>(sorted.data<char>());
  // Sorted by magnitude: |1|, |3|, |4|, |5|
  EXPECT_NEAR(std::abs(d[0]), 1.0, 1e-10);
  EXPECT_NEAR(std::abs(d[1]), 3.0, 1e-10);
  EXPECT_NEAR(std::abs(d[2]), 4.0, 1e-10);
  EXPECT_NEAR(std::abs(d[3]), 5.0, 1e-10);
}

TEST_F(SignalFilterDesignTestCPU, CmplxSortSingle) {
  std::vector<double> data = {42.0};
  Array arr = to_array(data);
  Array sorted = signal::cmplx_sort(arr);
  EXPECT_NEAR(sorted.data<double>()[0], 42.0, 1e-10);
}

TEST_F(SignalFilterDesignTestCPU, CmplxSortZeros) {
  std::vector<double> data = {0.0, 0.0, 0.0};
  Array arr = to_array(data);
  Array sorted = signal::cmplx_sort(arr);
  EXPECT_NEAR(sorted.data<double>()[0], 0.0, 1e-10);
  EXPECT_NEAR(sorted.data<double>()[1], 0.0, 1e-10);
  EXPECT_NEAR(sorted.data<double>()[2], 0.0, 1e-10);
}
