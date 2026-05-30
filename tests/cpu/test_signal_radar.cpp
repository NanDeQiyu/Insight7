// tests/cpu/test_signal_radar.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;
using namespace ins::signal;

class RadarTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
  }
};

// ============================================================================
// cfar_alpha
// ============================================================================

TEST_F(RadarTestCPU, CfarAlphaBasic) {
  double alpha = cfar_alpha(1e-3, 20);
  // alpha = 20 * (1e-3^(-1/20) - 1) ≈ 20 * (1.4125 - 1) ≈ 8.25
  EXPECT_GT(alpha, 5.0);
  EXPECT_LT(alpha, 15.0);
}

TEST_F(RadarTestCPU, CfarAlphaSmallPfa) {
  double alpha = cfar_alpha(1e-6, 100);
  EXPECT_GT(alpha, 0.0);
}

// ============================================================================
// ca_cfar 1D
// ============================================================================

TEST_F(RadarTestCPU, CaCfar1DBasic) {
  // Create signal with a target at index 50
  int N = 100;
  std::vector<double> data(N, 1.0);
  data[50] = 100.0; // target

  Array arr = to_array(data, DType::F64, CPUPlace());
  auto [threshold, detections] = ca_cfar(arr, {2}, {5}, 1e-3);

  EXPECT_EQ(threshold.numel(), N);
  EXPECT_EQ(detections.numel(), N);

  // Target should be detected
  const bool *det = detections.data<bool>();
  EXPECT_TRUE(det[50]);

  // Noise region should not be detected (mostly)
  int false_alarms = 0;
  for (int i = 0; i < N; ++i) {
    if (i >= 40 && i <= 60)
      continue;
    if (det[i])
      false_alarms++;
  }
  EXPECT_LT(false_alarms, 10); // very few false alarms
}

TEST_F(RadarTestCPU, CaCfar1DF32) {
  std::vector<float> data(100, 1.0f);
  data[50] = 100.0f;
  Array arr = to_array(data, DType::F32, CPUPlace());

  auto [threshold, detections] = ca_cfar(arr, {2}, {5}, 1e-3);

  EXPECT_EQ(threshold.dtype(), DType::F32);
  EXPECT_TRUE(detections.data<bool>()[50]);
}

TEST_F(RadarTestCPU, CaCfar1DNoTarget) {
  std::vector<double> data(100, 1.0);
  Array arr = to_array(data, DType::F64, CPUPlace());

  auto [threshold, detections] = ca_cfar(arr, {2}, {5}, 1e-3);

  // Very few detections in uniform noise
  const bool *det = detections.data<bool>();
  int count = 0;
  for (int i = 0; i < 100; ++i) {
    if (det[i])
      count++;
  }
  EXPECT_LT(count, 10);
}

// ============================================================================
// ca_cfar 2D
// ============================================================================

TEST_F(RadarTestCPU, CaCfar2DBasic) {
  // 20x20 array with target at (10, 10)
  int R = 20, C = 20;
  std::vector<double> data(R * C, 1.0);
  data[10 * C + 10] = 100.0;

  Array arr = to_array(data, Shape({R, C}), DType::F64, CPUPlace());
  auto [threshold, detections] = ca_cfar(arr, {1, 1}, {2, 2}, 1e-3);

  EXPECT_EQ(threshold.numel(), R * C);
  const bool *det = detections.data<bool>();
  EXPECT_TRUE(det[10 * C + 10]);
}

// ============================================================================
// mvdr
// ============================================================================

TEST_F(RadarTestCPU, MvdrBasic) {
  // Simple 2-sensor, 10-sample case with independent data
  int M = 2, N = 10;
  std::vector<double> x_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
  Array x = to_array(x_data, Shape({M, N}), DType::F64, CPUPlace());

  std::vector<double> sv_data = {1.0, 0.5};
  Array sv = to_array(sv_data, DType::F64, CPUPlace());

  Array w = mvdr(x, sv, true);

  EXPECT_EQ(w.numel(), M);
  EXPECT_TRUE(std::isfinite(w.data<double>()[0]));
  EXPECT_TRUE(std::isfinite(w.data<double>()[1]));
}

TEST_F(RadarTestCPU, MvdrInvalidShape) {
  // More sensors than samples should fail
  std::vector<double> x_data = {1, 2, 3, 4, 5, 6};
  Array x = to_array(x_data, Shape({3, 2}), DType::F64, CPUPlace());
  std::vector<double> sv_data = {1.0, 0.5, 0.3};
  Array sv = to_array(sv_data, DType::F64, CPUPlace());

  EXPECT_THROW(mvdr(x, sv, true), std::exception);
}

// ============================================================================
// pulse_compression
// ============================================================================

TEST_F(RadarTestCPU, PulseCompressionBasic) {
  // 3 pulses, 8 samples each
  int num_pulses = 3, samples = 8;
  std::vector<double> x_data(num_pulses * samples);
  for (int i = 0; i < num_pulses * samples; ++i)
    x_data[i] = std::sin(2.0 * M_PI * i / 8.0);

  // Template: 4 samples
  std::vector<double> tpl_data = {1.0, 0.5, -0.5, -1.0};

  Array x =
      to_array(x_data, Shape({num_pulses, samples}), DType::F64, CPUPlace());
  Array tpl = to_array(tpl_data, DType::F64, CPUPlace());

  Array result = pulse_compression(x, tpl, false, "", 0);

  EXPECT_EQ(result.shape().dim(0), num_pulses);
  EXPECT_EQ(result.shape().dim(1), samples);
  EXPECT_EQ(result.dtype(), DType::C64);
}

TEST_F(RadarTestCPU, PulseCompressionNormalize) {
  int num_pulses = 2, samples = 8;
  std::vector<double> x_data(num_pulses * samples, 1.0);
  std::vector<double> tpl_data = {1.0, 1.0, 1.0, 1.0};

  Array x =
      to_array(x_data, Shape({num_pulses, samples}), DType::F64, CPUPlace());
  Array tpl = to_array(tpl_data, DType::F64, CPUPlace());

  Array result = pulse_compression(x, tpl, true);

  EXPECT_EQ(result.shape().dim(0), num_pulses);
}

// ============================================================================
// pulse_doppler
// ============================================================================

TEST_F(RadarTestCPU, PulseDopplerBasic) {
  int num_pulses = 8, samples = 4;
  std::vector<double> x_data(num_pulses * samples);
  for (int i = 0; i < num_pulses * samples; ++i)
    x_data[i] = std::sin(2.0 * M_PI * i / 8.0);

  Array x =
      to_array(x_data, Shape({num_pulses, samples}), DType::F64, CPUPlace());

  Array result = pulse_doppler(x, "", 0);

  EXPECT_EQ(result.shape().dim(0), num_pulses);
  EXPECT_EQ(result.shape().dim(1), samples);
  EXPECT_EQ(result.dtype(), DType::C64);
}
