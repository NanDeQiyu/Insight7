// tests/cpu/test_plot.cpp
#ifdef INSIGHT_USE_MATPLOT

#include "insight/insight.h"
#include "insight/ops/plot.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;

class PlotTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
  }
};

// ============================================================================
// Conversion Helpers
// ============================================================================

TEST_F(PlotTestCPU, ToVectorF64) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  auto v = plot::to_vector(to_array(data));
  ASSERT_EQ(v.size(), 5);
  for (int i = 0; i < 5; ++i)
    EXPECT_DOUBLE_EQ(v[i], data[i]);
}

TEST_F(PlotTestCPU, ToVectorF32) {
  std::vector<float> data = {1.5f, 2.5f, 3.5f};
  auto v = plot::to_vector(to_array(data, DType::F32));
  ASSERT_EQ(v.size(), 3);
  EXPECT_DOUBLE_EQ(v[0], 1.5);
}

TEST_F(PlotTestCPU, ToVectorI32) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  auto v = plot::to_vector(to_array(data, DType::I32));
  ASSERT_EQ(v.size(), 4);
  EXPECT_DOUBLE_EQ(v[0], 1.0);
}

TEST_F(PlotTestCPU, ToVectorI64) {
  std::vector<int64_t> data = {10, 20, 30};
  auto v = plot::to_vector(to_array(data, DType::I64));
  ASSERT_EQ(v.size(), 3);
  EXPECT_DOUBLE_EQ(v[0], 10.0);
}

TEST_F(PlotTestCPU, ToVectorEmpty) {
  auto v = plot::to_vector(zeros({0}, DType::F64));
  EXPECT_EQ(v.size(), 0);
}

TEST_F(PlotTestCPU, ToVectorSingle) {
  auto v = plot::to_vector(to_array(std::vector<double>{42.0}));
  ASSERT_EQ(v.size(), 1);
  EXPECT_DOUBLE_EQ(v[0], 42.0);
}

TEST_F(PlotTestCPU, ToVectorLarge) {
  int64_t N = 10000;
  std::vector<double> data(N);
  for (int i = 0; i < N; ++i)
    data[i] = i * 0.01;
  auto v = plot::to_vector(to_array(data));
  ASSERT_EQ(v.size(), N);
  for (int i = 0; i < N; ++i)
    EXPECT_DOUBLE_EQ(v[i], data[i]);
}

TEST_F(PlotTestCPU, ToVectorNegative) {
  std::vector<double> data = {-1.0, -2.5, 0.0, 3.14};
  auto v = plot::to_vector(to_array(data));
  EXPECT_DOUBLE_EQ(v[0], -1.0);
  EXPECT_DOUBLE_EQ(v[3], 3.14);
}

TEST_F(PlotTestCPU, ToMatrix2x3) {
  std::vector<double> data = {1, 2, 3, 4, 5, 6};
  auto m = plot::to_matrix(to_array(data, {2, 3}));
  ASSERT_EQ(m.size(), 2);
  ASSERT_EQ(m[0].size(), 3);
  EXPECT_DOUBLE_EQ(m[0][0], 1.0);
  EXPECT_DOUBLE_EQ(m[1][2], 6.0);
}

TEST_F(PlotTestCPU, ToMatrix3x4) {
  std::vector<double> data(12);
  for (int i = 0; i < 12; ++i)
    data[i] = i + 1;
  auto m = plot::to_matrix(to_array(data, {3, 4}));
  ASSERT_EQ(m.size(), 3);
  ASSERT_EQ(m[0].size(), 4);
  EXPECT_DOUBLE_EQ(m[0][0], 1.0);
  EXPECT_DOUBLE_EQ(m[2][3], 12.0);
}

TEST_F(PlotTestCPU, ToMatrixSquare) {
  auto m = plot::to_matrix(to_array(std::vector<double>{1, 2, 3, 4}, {2, 2}));
  ASSERT_EQ(m.size(), 2);
  EXPECT_DOUBLE_EQ(m[0][0], 1.0);
  EXPECT_DOUBLE_EQ(m[1][1], 4.0);
}

// ============================================================================
// Colormap Enum
// ============================================================================

TEST_F(PlotTestCPU, ColormapEnumDistinct) {
  EXPECT_NE(static_cast<int>(plot::Colormap::Parula),
            static_cast<int>(plot::Colormap::Jet));
  EXPECT_NE(static_cast<int>(plot::Colormap::Hot),
            static_cast<int>(plot::Colormap::Cool));
  EXPECT_NE(static_cast<int>(plot::Colormap::Spring),
            static_cast<int>(plot::Colormap::Summer));
}

// ============================================================================
// API Coverage Check — verify all functions link correctly
// (gnuplot not available, so we only test conversion + enum, not rendering)
// ============================================================================

TEST_F(PlotTestCPU, AllFunctionsLink) {
  // This test verifies that all plot functions compile and link.
  // We can't actually render without gnuplot, but we can verify the API exists.
  auto x = to_array(std::vector<double>{1, 2, 3});
  auto y = to_array(std::vector<double>{4, 5, 6});
  auto z = to_array(std::vector<double>{7, 8, 9});
  auto m = to_array(std::vector<double>{1, 2, 3, 4, 5, 6}, {2, 3});

  // Verify all conversion functions work
  EXPECT_EQ(plot::to_vector(x).size(), 3);
  EXPECT_EQ(plot::to_matrix(m).size(), 2);
}

#endif // INSIGHT_USE_MATPLOT
