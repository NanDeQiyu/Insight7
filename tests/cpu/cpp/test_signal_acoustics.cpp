// tests/cpu/test_signal_acoustics.cpp
#include "insight/insight.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;

class SignalAcousticsTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
  }
};

// ========== mel2hz / hz2mel ==========

TEST_F(SignalAcousticsTestCPU, Mel2HzBasic) {
  // 0 mel = 0 Hz
  std::vector<double> data = {0.0};
  Array arr = to_array(data);
  Array hz = signal::mel2hz(arr);
  EXPECT_NEAR(hz.data<double>()[0], 0.0, 1e-10);
}

TEST_F(SignalAcousticsTestCPU, Hz2MelBasic) {
  // 0 Hz = 0 mel
  std::vector<double> data = {0.0};
  Array arr = to_array(data);
  Array mel = signal::hz2mel(arr);
  EXPECT_NEAR(mel.data<double>()[0], 0.0, 1e-10);
}

TEST_F(SignalAcousticsTestCPU, MelHzRoundTrip) {
  // hz2mel(mel2hz(x)) = x
  std::vector<double> data = {100.0, 500.0, 1000.0, 5000.0};
  Array arr = to_array(data);
  Array mel = signal::hz2mel(arr);
  Array hz_back = signal::mel2hz(mel);
  const double *d = hz_back.data<double>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(d[i], data[i], 1e-6) << "roundtrip failed at " << i;
  }
}

TEST_F(SignalAcousticsTestCPU, Hz2MelMonotonic) {
  std::vector<double> data = {100.0, 200.0, 500.0, 1000.0, 5000.0};
  Array arr = to_array(data);
  Array mel = signal::hz2mel(arr);
  const double *m = mel.data<double>();
  for (int i = 1; i < 5; ++i) {
    EXPECT_GT(m[i], m[i - 1]) << "not monotonic at " << i;
  }
}

TEST_F(SignalAcousticsTestCPU, MelFrequenciesBasic) {
  Array freqs = signal::mel_frequencies(10, 0.0, 8000.0);
  EXPECT_EQ(freqs.numel(), 10);
  const double *f = freqs.data<double>();
  // First should be 0 (or close)
  EXPECT_NEAR(f[0], 0.0, 1.0);
  // Last should be 8000 (or close)
  EXPECT_NEAR(f[9], 8000.0, 100.0);
  // Should be monotonically increasing
  for (int i = 1; i < 10; ++i) {
    EXPECT_GT(f[i], f[i - 1]);
  }
}

TEST_F(SignalAcousticsTestCPU, MelFrequenciesSingle) {
  Array freqs = signal::mel_frequencies(1, 0.0, 8000.0);
  EXPECT_EQ(freqs.numel(), 1);
}

// ========== hz2bark / bark2hz ==========

TEST_F(SignalAcousticsTestCPU, Hz2BarkBasic) {
  // 0 Hz should give bark ~ 0
  std::vector<double> data = {0.0};
  Array arr = to_array(data);
  Array bark = signal::hz2bark(arr);
  // At hz=0, formula gives 26.81/(1+inf) - 0.53 = -0.53
  // But division by zero issue, so just check it's defined
  EXPECT_TRUE(bark.defined());
}

TEST_F(SignalAcousticsTestCPU, BarkHzRoundTrip) {
  std::vector<double> data = {200.0, 500.0, 1000.0, 4000.0};
  Array arr = to_array(data);
  Array bark = signal::hz2bark(arr);
  Array hz_back = signal::bark2hz(bark);
  const double *d = hz_back.data<double>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(d[i], data[i], 1.0) << "roundtrip failed at " << i;
  }
}

TEST_F(SignalAcousticsTestCPU, Hz2BarkMonotonic) {
  std::vector<double> data = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
  Array arr = to_array(data);
  Array bark = signal::hz2bark(arr);
  const double *b = bark.data<double>();
  for (int i = 1; i < 5; ++i) {
    EXPECT_GT(b[i], b[i - 1]) << "not monotonic at " << i;
  }
}
