// tests/cpu/test_signal_waveforms.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

class SignalWaveformsTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
  }
};

namespace {
template <typename T>
void expect_float_equal(const Array &arr, const std::vector<T> &expected,
                        T tol = 1e-5) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_NEAR(data[i], expected[i], tol) << "mismatch at index " << i;
  }
}
} // namespace

// ========== sawtooth ==========

TEST_F(SignalWaveformsTestCPU, SawtoothBasic) {
  // sawtooth(0, width=1.0) should be -1 (start of ramp)
  // sawtooth(pi, width=1.0) should be 0 (midpoint)
  // sawtooth(2*pi - eps, width=1.0) should be ~1 (end of ramp)
  std::vector<double> t_data = {0.0, M_PI, 2.0 * M_PI - 1e-10};
  Array t = to_array(t_data);
  Array y = signal::sawtooth(t, 1.0);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], -1.0, 1e-6);
  EXPECT_NEAR(d[1], 0.0, 1e-6);
  EXPECT_NEAR(d[2], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestCPU, SawtoothTriangle) {
  // width=0.5 gives triangle wave
  // sawtooth(0, 0.5) = -1
  // sawtooth(pi/2, 0.5) = 0 (peak of triangle at pi for width=0.5)
  // sawtooth(pi, 0.5) = 1
  std::vector<double> t_data = {0.0, M_PI / 2.0, M_PI};
  Array t = to_array(t_data);
  Array y = signal::sawtooth(t, 0.5);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], -1.0, 1e-6);
  EXPECT_NEAR(d[1], 0.0, 1e-6);
  EXPECT_NEAR(d[2], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestCPU, SawtoothPeriodicity) {
  // sawtooth should be periodic with period 2*pi
  std::vector<double> t_data = {0.5, 0.5 + 2.0 * M_PI, 0.5 + 4.0 * M_PI};
  Array t = to_array(t_data);
  Array y = signal::sawtooth(t, 1.0);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], d[1], 1e-6);
  EXPECT_NEAR(d[0], d[2], 1e-6);
}

// ========== square ==========

TEST_F(SignalWaveformsTestCPU, SquareBasic) {
  // square(0, duty=0.5) = +1 (start of high region)
  // square(pi, duty=0.5) = -1 (second half of period)
  // square(2*pi - eps, duty=0.5) = -1
  std::vector<double> t_data = {0.0, M_PI, 2.0 * M_PI - 1e-10};
  Array t = to_array(t_data);
  Array y = signal::square(t, 0.5);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-6);
  EXPECT_NEAR(d[1], -1.0, 1e-6);
  EXPECT_NEAR(d[2], -1.0, 1e-6);
}

TEST_F(SignalWaveformsTestCPU, SquareDuty) {
  // duty=0.25: high for first quarter, low for rest
  std::vector<double> t_data = {0.0, M_PI / 2.0 - 1e-10, M_PI / 2.0 + 1e-10};
  Array t = to_array(t_data);
  Array y = signal::square(t, 0.25);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-6);
  EXPECT_NEAR(d[1], 1.0, 1e-6);
  EXPECT_NEAR(d[2], -1.0, 1e-6);
}

TEST_F(SignalWaveformsTestCPU, SquarePeriodicity) {
  std::vector<double> t_data = {1.0, 1.0 + 2.0 * M_PI, 1.0 + 4.0 * M_PI};
  Array t = to_array(t_data);
  Array y = signal::square(t, 0.5);
  const double *d = y.data<double>();
  EXPECT_NEAR(d[0], d[1], 1e-6);
  EXPECT_NEAR(d[0], d[2], 1e-6);
}

// ========== gausspulse ==========

TEST_F(SignalWaveformsTestCPU, GaussPulsePeak) {
  // gausspulse should peak at t=0 (envelope = 1, cos(0) = 1)
  std::vector<double> t_data = {0.0};
  Array t = to_array(t_data);
  Array y = signal::gausspulse(t, 1000.0);
  EXPECT_NEAR(y.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestCPU, GaussPulseDecay) {
  // gausspulse should decay away from t=0
  std::vector<double> t_data = {0.0, 0.001, 0.01};
  Array t = to_array(t_data);
  Array y = signal::gausspulse(t, 1000.0, 0.5);
  const double *d = y.data<double>();
  // Peak at t=0
  EXPECT_NEAR(d[0], 1.0, 1e-6);
  // Should decay as |t| increases
  EXPECT_GT(std::abs(d[0]), std::abs(d[2]));
}

TEST_F(SignalWaveformsTestCPU, GaussPulseFull) {
  std::vector<double> t_data = {0.0, 0.0005};
  Array t = to_array(t_data);
  auto result = signal::gausspulse_full(t, 1000.0, 0.5);
  // At t=0: yi = env*cos(0) = 1, yq = env*sin(0) = 0, env = 1
  EXPECT_NEAR(result.yi.data<double>()[0], 1.0, 1e-6);
  EXPECT_NEAR(result.yq.data<double>()[0], 0.0, 1e-6);
  EXPECT_NEAR(result.ye.data<double>()[0], 1.0, 1e-6);
}

// ========== chirp ==========

TEST_F(SignalWaveformsTestCPU, ChirpLinear) {
  // At t=0: phase = 2*pi*f0*0 + phi = phi
  // cos(phi) with default phi=0 → cos(0) = 1
  std::vector<double> t_data = {0.0};
  Array t = to_array(t_data);
  Array y = signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Linear);
  EXPECT_NEAR(y.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestCPU, ChirpQuadratic) {
  // At t=0: same as linear → cos(phi) = 1
  std::vector<double> t_data = {0.0};
  Array t = to_array(t_data);
  Array y = signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Quadratic,
                          0.0, true);
  EXPECT_NEAR(y.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestCPU, ChirpLogarithmic) {
  std::vector<double> t_data = {0.0};
  Array t = to_array(t_data);
  Array y = signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Logarithmic);
  EXPECT_NEAR(y.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestCPU, ChirpHyperbolic) {
  std::vector<double> t_data = {0.0};
  Array t = to_array(t_data);
  Array y = signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Hyperbolic);
  EXPECT_NEAR(y.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestCPU, ChirpLinearEndFrequency) {
  // For linear chirp, at t=t1 the instantaneous frequency should be f1
  // We verify by checking the phase derivative indirectly:
  // phase(t) = 2*pi*(f0*t + 0.5*beta*t^2), dphase/dt = 2*pi*(f0 + beta*t)
  // At t=t1: dphase/dt = 2*pi*(f0 + (f1-f0)/t1 * t1) = 2*pi*f1
  // So cos(phase(t1)) = cos(2*pi*(f0*t1 + 0.5*(f1-f0)*t1))
  // For f0=6, t1=1, f1=10: cos(2*pi*(6 + 2)) = cos(16*pi) = 1
  std::vector<double> t_data = {1.0};
  Array t = to_array(t_data);
  Array y = signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Linear);
  EXPECT_NEAR(y.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestCPU, ChirpWithPhaseOffset) {
  // With phi=pi/2: cos(0 + pi/2) = 0
  std::vector<double> t_data = {0.0};
  Array t = to_array(t_data);
  Array y =
      signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Linear, M_PI / 2.0);
  EXPECT_NEAR(y.data<double>()[0], 0.0, 1e-6);
}

// ========== unit_impulse ==========

TEST_F(SignalWaveformsTestCPU, UnitImpulseCenter) {
  // Default idx=-1 means center
  Array y = signal::unit_impulse(Shape({11}));
  EXPECT_EQ(y.numel(), 11);
  const double *d = y.data<double>();
  for (int64_t i = 0; i < 11; ++i) {
    if (i == 5) {
      EXPECT_NEAR(d[i], 1.0, 1e-10);
    } else {
      EXPECT_NEAR(d[i], 0.0, 1e-10);
    }
  }
}

TEST_F(SignalWaveformsTestCPU, UnitImpulseAtIdx) {
  Array y = signal::unit_impulse(Shape({7}), 3);
  const double *d = y.data<double>();
  for (int64_t i = 0; i < 7; ++i) {
    if (i == 3) {
      EXPECT_NEAR(d[i], 1.0, 1e-10);
    } else {
      EXPECT_NEAR(d[i], 0.0, 1e-10);
    }
  }
}

TEST_F(SignalWaveformsTestCPU, UnitImpulseFloat32) {
  Array y = signal::unit_impulse(Shape({5}), 2, DType::F32);
  EXPECT_EQ(y.dtype(), DType::F32);
  const float *d = y.data<float>();
  for (int64_t i = 0; i < 5; ++i) {
    if (i == 2) {
      EXPECT_FLOAT_EQ(d[i], 1.0f);
    } else {
      EXPECT_FLOAT_EQ(d[i], 0.0f);
    }
  }
}
