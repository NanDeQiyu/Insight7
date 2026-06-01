// tests/cpu/test_signal_convolution.cpp
#include "insight/insight.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;

class SignalConvolutionTestCPU : public ::testing::Test {
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

// ========== fftconvolve ==========

TEST_F(SignalConvolutionTestCPU, FftconvolveBasic) {
  // [1, 2, 3] * [1, 1] = [1, 3, 5, 3]
  std::vector<double> a = {1, 2, 3};
  std::vector<double> b = {1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c = signal::fftconvolve(A, B, "full");
  expect_float_equal(c, std::vector<double>{1, 3, 5, 3}, 1e-10);
}

TEST_F(SignalConvolutionTestCPU, FftconvolveSame) {
  std::vector<double> a = {1, 2, 3, 4, 5};
  std::vector<double> b = {1, 1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c = signal::fftconvolve(A, B, "same");
  // full = [1, 3, 6, 9, 12, 9, 5], same = center 5 = [3, 6, 9, 12, 9]
  expect_float_equal(c, std::vector<double>{3, 6, 9, 12, 9}, 1e-10);
}

TEST_F(SignalConvolutionTestCPU, FftconvolveValid) {
  std::vector<double> a = {1, 2, 3, 4, 5};
  std::vector<double> b = {1, 1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c = signal::fftconvolve(A, B, "valid");
  // valid = [6, 9, 12]
  expect_float_equal(c, std::vector<double>{6, 9, 12}, 1e-10);
}

TEST_F(SignalConvolutionTestCPU, FftconvolveCommutative) {
  std::vector<double> a = {1, 2, 3};
  std::vector<double> b = {4, 5};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c1 = signal::fftconvolve(A, B, "full");
  Array c2 = signal::fftconvolve(B, A, "full");
  const double *d1 = c1.data<double>();
  const double *d2 = c2.data<double>();
  ASSERT_EQ(c1.numel(), c2.numel());
  for (int64_t i = 0; i < c1.numel(); ++i) {
    EXPECT_NEAR(d1[i], d2[i], 1e-10);
  }
}

TEST_F(SignalConvolutionTestCPU, FftconvolveImpulse) {
  // Convolution with impulse should return the other signal
  std::vector<double> a = {1, 2, 3, 4, 5};
  std::vector<double> b = {1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c = signal::fftconvolve(A, B, "full");
  expect_float_equal(c, std::vector<double>{1, 2, 3, 4, 5}, 1e-10);
}

TEST_F(SignalConvolutionTestCPU, Fftconvolve2D) {
  // 2D convolution: 3x3 * 2x2
  std::vector<double> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b_data = {1, 0, 0, 1};
  Array A = to_array(a_data, {3, 3});
  Array B = to_array(b_data, {2, 2});
  Array c = signal::fftconvolve(A, B, "full");
  // Full 2D convolution output should be 4x4
  EXPECT_EQ(c.shape().dim(0), 4);
  EXPECT_EQ(c.shape().dim(1), 4);
  // c[0,0] = 1*1 = 1
  EXPECT_NEAR(c.data<double>()[0], 1.0, 1e-10);
  // c[1,1] = 1*5 + 0*4 + 0*2 + 1*1 = 5+1 = 6
  EXPECT_NEAR(c.data<double>()[5], 6.0, 1e-10);
}

TEST_F(SignalConvolutionTestCPU, Fftconvolve2DSame) {
  std::vector<double> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b_data = {1, 0, 0, 1};
  Array A = to_array(a_data, {3, 3});
  Array B = to_array(b_data, {2, 2});
  Array c = signal::fftconvolve(A, B, "same");
  EXPECT_EQ(c.shape().dim(0), 3);
  EXPECT_EQ(c.shape().dim(1), 3);
}

// ========== correlate ==========

TEST_F(SignalConvolutionTestCPU, CorrelateBasic) {
  // Cross-correlation of [1, 2, 3] with [1, 1]
  // = convolve([1,2,3], [1,1] reversed) = convolve([1,2,3], [1,1])
  // = [1, 3, 5, 3]
  std::vector<double> a = {1, 2, 3};
  std::vector<double> b = {1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c = signal::correlate(A, B, "full");
  expect_float_equal(c, std::vector<double>{1, 3, 5, 3}, 1e-10);
}

TEST_F(SignalConvolutionTestCPU, CorrelateSame) {
  std::vector<double> a = {1, 2, 3, 4, 5};
  std::vector<double> b = {1, 1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c = signal::correlate(A, B, "same");
  // correlate with [1,1,1] reversed = [1,1,1] (symmetric)
  // Same as convolve with [1,1,1]
  expect_float_equal(c, std::vector<double>{3, 6, 9, 12, 9}, 1e-10);
}

TEST_F(SignalConvolutionTestCPU, CorrelateValid) {
  std::vector<double> a = {1, 2, 3, 4, 5};
  std::vector<double> b = {1, 1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c = signal::correlate(A, B, "valid");
  expect_float_equal(c, std::vector<double>{6, 9, 12}, 1e-10);
}

TEST_F(SignalConvolutionTestCPU, CorrelateAutoCorrelation) {
  // Auto-correlation of [1, 0, -1]: should be symmetric
  std::vector<double> a = {1, 0, -1};
  Array A = to_array(a);
  Array c = signal::correlate(A, A, "full");
  // [1,0,-1] * [1,0,-1] reversed = [1,0,-1]*[-1,0,1]
  // = [-1, 0, 0, 0, -1] ... let me compute:
  // full cross-correlation: sum of a[i]*a[i+k]
  // k=-2: 1*(-1) = -1
  // k=-1: 1*0 + 0*(-1) = 0
  // k=0: 1*1 + 0*0 + (-1)*(-1) = 2
  // k=1: 0*1 + (-1)*0 = 0
  // k=2: (-1)*1 = -1
  const double *d = c.data<double>();
  ASSERT_EQ(c.numel(), 5);
  EXPECT_NEAR(d[0], -1.0, 1e-10);
  EXPECT_NEAR(d[1], 0.0, 1e-10);
  EXPECT_NEAR(d[2], 2.0, 1e-10);
  EXPECT_NEAR(d[3], 0.0, 1e-10);
  EXPECT_NEAR(d[4], -1.0, 1e-10);
}

// ========== convolve2d ==========

TEST_F(SignalConvolutionTestCPU, Convolve2dBasic) {
  std::vector<double> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b_data = {1, 0, 0, 1};
  Array A = to_array(a_data, {3, 3});
  Array B = to_array(b_data, {2, 2});
  Array c = signal::convolve2d(A, B, "full");
  EXPECT_EQ(c.shape().dim(0), 4);
  EXPECT_EQ(c.shape().dim(1), 4);
}

TEST_F(SignalConvolutionTestCPU, Convolve2dValid) {
  // Use a simpler case: 3x3 * 3x3 = 5x5 full, 1x1 valid
  std::vector<double> a_data = {1, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> b_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Array A = to_array(a_data, {3, 3});
  Array B = to_array(b_data, {3, 3});
  Array c = signal::convolve2d(A, B, "full");
  // A is identity-like, so full conv should be B padded with zeros
  EXPECT_EQ(c.shape().dim(0), 5);
  EXPECT_EQ(c.shape().dim(1), 5);
  // c[0,0] = 1*1 = 1
  EXPECT_NEAR(c.data<double>()[0], 1.0, 1e-10);

  // Test same mode (3x3 output)
  Array cs = signal::convolve2d(A, B, "same");
  EXPECT_EQ(cs.shape().dim(0), 3);
  EXPECT_EQ(cs.shape().dim(1), 3);

  // Test with 2x2 kernel on 3x3 input
  std::vector<double> k_data = {1, 1, 1, 1};
  Array K = to_array(k_data, {2, 2});
  Array cv = signal::convolve2d(A, K, "valid");
  EXPECT_EQ(cv.shape().dim(0), 2);
  EXPECT_EQ(cv.shape().dim(1), 2);
  // A = [[1,0,0],[0,0,0],[0,0,0]], K = [[1,1],[1,1]]
  // valid[0,0] = 1*1+0*1+0*1+0*1 = 1
  // valid[0,1] = 0*1+0*1+0*1+0*1 = 0
  // valid[1,0] = 0*1+0*1+0*1+0*1 = 0
  // valid[1,1] = 0*1+0*1+0*1+0*1 = 0
  EXPECT_NEAR(cv.data<double>()[0], 1.0, 1e-10);
  EXPECT_NEAR(cv.data<double>()[1], 0.0, 1e-10);
  EXPECT_NEAR(cv.data<double>()[2], 0.0, 1e-10);
  EXPECT_NEAR(cv.data<double>()[3], 0.0, 1e-10);
}

// ========== correlate2d ==========

TEST_F(SignalConvolutionTestCPU, Correlate2dBasic) {
  std::vector<double> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b_data = {1, 1, 1, 1};
  Array A = to_array(a_data, {3, 3});
  Array B = to_array(b_data, {2, 2});
  Array c = signal::correlate2d(A, B, "valid");
  EXPECT_EQ(c.shape().dim(0), 2);
  EXPECT_EQ(c.shape().dim(1), 2);
}

TEST_F(SignalConvolutionTestCPU, Correlate2dFull) {
  std::vector<double> a_data = {1, 2, 3, 4};
  std::vector<double> b_data = {1, 1, 1, 1};
  Array A = to_array(a_data, {2, 2});
  Array B = to_array(b_data, {2, 2});
  Array c = signal::correlate2d(A, B, "full");
  EXPECT_EQ(c.shape().dim(0), 3);
  EXPECT_EQ(c.shape().dim(1), 3);
}

// ========== choose_conv_method ==========

TEST_F(SignalConvolutionTestCPU, ChooseConvMethodSmall) {
  std::vector<double> a = {1, 2, 3};
  std::vector<double> b = {1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  std::string method = signal::choose_conv_method(A, B, "full");
  EXPECT_EQ(method, "direct");
}

TEST_F(SignalConvolutionTestCPU, ChooseConvMethodLarge) {
  // Large arrays should prefer FFT
  std::vector<double> a(1000, 1.0);
  std::vector<double> b(1000, 1.0);
  Array A = to_array(a);
  Array B = to_array(b);
  std::string method = signal::choose_conv_method(A, B, "full");
  EXPECT_EQ(method, "fft");
}

// ========== correlation_lags ==========

TEST_F(SignalConvolutionTestCPU, CorrelationLagsFull) {
  Array lags = signal::correlation_lags(5, 3, "full");
  // Full lags: -(3-1) to (5-1) = -2 to 4
  ASSERT_EQ(lags.numel(), 7);
  const int64_t *d = lags.data<int64_t>();
  EXPECT_EQ(d[0], -2);
  EXPECT_EQ(d[1], -1);
  EXPECT_EQ(d[2], 0);
  EXPECT_EQ(d[3], 1);
  EXPECT_EQ(d[4], 2);
  EXPECT_EQ(d[5], 3);
  EXPECT_EQ(d[6], 4);
}

TEST_F(SignalConvolutionTestCPU, CorrelationLagsSame) {
  Array lags = signal::correlation_lags(5, 3, "same");
  ASSERT_EQ(lags.numel(), 5);
  const int64_t *d = lags.data<int64_t>();
  EXPECT_EQ(d[0], -1);
  EXPECT_EQ(d[4], 3);
}

TEST_F(SignalConvolutionTestCPU, CorrelationLagsValid) {
  Array lags = signal::correlation_lags(5, 3, "valid");
  // valid: where both arrays fully overlap
  // in1_len=5, in2_len=3, valid output len = 5-3+1 = 3
  ASSERT_EQ(lags.numel(), 3);
  const int64_t *d = lags.data<int64_t>();
  EXPECT_EQ(d[0], 0);
  EXPECT_EQ(d[2], 2);
}

TEST_F(SignalConvolutionTestCPU, CorrelationLagsEqual) {
  Array lags = signal::correlation_lags(3, 3, "full");
  ASSERT_EQ(lags.numel(), 5);
  const int64_t *d = lags.data<int64_t>();
  EXPECT_EQ(d[0], -2);
  EXPECT_EQ(d[4], 2);
}
