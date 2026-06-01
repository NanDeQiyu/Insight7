// tests/cpu/test_signal_bsplines.cpp
#include "insight/insight.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;

class SignalBsplinesTestCPU : public ::testing::Test {
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

// ========== gauss_spline ==========

TEST_F(SignalBsplinesTestCPU, GaussSplineBasic) {
  // At x=0: gauss_spline(0, n) = 1/sqrt(2*pi*sigma^2) where sigma^2=(n+1)/12
  std::vector<double> x_data = {0.0};
  Array x = to_array(x_data);
  Array y = signal::gauss_spline(x, 3);
  double sigma_sq = 4.0 / 12.0;
  double expected = 1.0 / std::sqrt(2.0 * M_PI * sigma_sq);
  EXPECT_NEAR(y.data<double>()[0], expected, 1e-10);
}

TEST_F(SignalBsplinesTestCPU, GaussSplineSymmetry) {
  std::vector<double> x_data = {-2.0, -1.0, 0.0, 1.0, 2.0};
  Array x = to_array(x_data);
  Array y = signal::gauss_spline(x, 2);
  const double *d = y.data<double>();
  // Symmetric around 0
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
  // Peak at 0
  EXPECT_GT(d[2], d[0]);
  EXPECT_GT(d[2], d[1]);
}

TEST_F(SignalBsplinesTestCPU, GaussSplineDecay) {
  // Should decay as |x| increases
  std::vector<double> x_data = {0.0, 1.0, 2.0, 3.0, 5.0};
  Array x = to_array(x_data);
  Array y = signal::gauss_spline(x, 3);
  const double *d = y.data<double>();
  for (int i = 1; i < 5; ++i) {
    EXPECT_LT(d[i], d[i - 1]);
  }
}

// ========== cubic ==========

TEST_F(SignalBsplinesTestCPU, CubicBasic) {
  // cubic(0) = 2/3
  std::vector<double> x_data = {0.0};
  Array x = to_array(x_data);
  Array y = signal::cubic(x);
  EXPECT_NEAR(y.data<double>()[0], 2.0 / 3.0, 1e-10);
}

TEST_F(SignalBsplinesTestCPU, CubicSymmetry) {
  std::vector<double> x_data = {-1.5, -0.5, 0.0, 0.5, 1.5};
  Array x = to_array(x_data);
  Array y = signal::cubic(x);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
}

TEST_F(SignalBsplinesTestCPU, CubicZeroOutside) {
  // cubic(x) = 0 for |x| >= 2
  std::vector<double> x_data = {-2.5, -2.0, 2.0, 2.5};
  Array x = to_array(x_data);
  Array y = signal::cubic(x);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[1], 0.0, 1e-10);
  EXPECT_NEAR(d[2], 0.0, 1e-10);
  EXPECT_NEAR(d[3], 0.0, 1e-10);
}

TEST_F(SignalBsplinesTestCPU, CubicRegion1) {
  // |x| < 1: 2/3 - 0.5*|x|^2*(2-|x|)
  // At x=0.5: 2/3 - 0.5*0.25*1.5 = 2/3 - 0.1875 = 0.479167
  std::vector<double> x_data = {0.5};
  Array x = to_array(x_data);
  Array y = signal::cubic(x);
  double expected = 2.0 / 3.0 - 0.5 * 0.25 * 1.5;
  EXPECT_NEAR(y.data<double>()[0], expected, 1e-10);
}

TEST_F(SignalBsplinesTestCPU, CubicRegion2) {
  // 1 <= |x| < 2: (1/6)*(2-|x|)^3
  // At x=1.5: (1/6)*(0.5)^3 = (1/6)*0.125 = 0.020833
  std::vector<double> x_data = {1.5};
  Array x = to_array(x_data);
  Array y = signal::cubic(x);
  double expected = (1.0 / 6.0) * 0.125;
  EXPECT_NEAR(y.data<double>()[0], expected, 1e-10);
}

// ========== quadratic ==========

TEST_F(SignalBsplinesTestCPU, QuadraticBasic) {
  // quadratic(0) = 0.75
  std::vector<double> x_data = {0.0};
  Array x = to_array(x_data);
  Array y = signal::quadratic(x);
  EXPECT_NEAR(y.data<double>()[0], 0.75, 1e-10);
}

TEST_F(SignalBsplinesTestCPU, QuadraticSymmetry) {
  std::vector<double> x_data = {-1.0, -0.25, 0.0, 0.25, 1.0};
  Array x = to_array(x_data);
  Array y = signal::quadratic(x);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
}

TEST_F(SignalBsplinesTestCPU, QuadraticZeroOutside) {
  // quadratic(x) = 0 for |x| >= 1.5
  std::vector<double> x_data = {-2.0, -1.5, 1.5, 2.0};
  Array x = to_array(x_data);
  Array y = signal::quadratic(x);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[1], 0.0, 1e-10);
  EXPECT_NEAR(d[2], 0.0, 1e-10);
  EXPECT_NEAR(d[3], 0.0, 1e-10);
}

TEST_F(SignalBsplinesTestCPU, QuadraticRegion1) {
  // |x| < 0.5: 0.75 - |x|^2
  std::vector<double> x_data = {0.3};
  Array x = to_array(x_data);
  Array y = signal::quadratic(x);
  EXPECT_NEAR(y.data<double>()[0], 0.75 - 0.09, 1e-10);
}

TEST_F(SignalBsplinesTestCPU, QuadraticRegion2) {
  // 0.5 <= |x| < 1.5: 0.5*(|x|-1.5)^2
  std::vector<double> x_data = {1.0};
  Array x = to_array(x_data);
  Array y = signal::quadratic(x);
  EXPECT_NEAR(y.data<double>()[0], 0.5 * 0.25, 1e-10);
}
