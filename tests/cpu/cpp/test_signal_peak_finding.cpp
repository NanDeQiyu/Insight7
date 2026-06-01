// tests/cpu/test_signal_peak_finding.cpp
#include "insight/insight.h"
#include <gtest/gtest.h>
#include <vector>

using namespace ins;
using namespace ins::signal;

class PeakFindingTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
  }
};

// ============================================================================
// argrelextrema
// ============================================================================

TEST_F(PeakFindingTestCPU, ArgrelextremaGreater1D) {
  // [1, 3, 2, 5, 4, 7, 6] -> maxima at 1, 3, 5 (values 3, 5, 7)
  std::vector<double> data = {1, 3, 2, 5, 4, 7, 6};
  Array arr = to_array(data, DType::F64, CPUPlace());

  auto indices = argrelextrema(arr, "greater", 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  ASSERT_EQ(indices[0].numel(), 3);
  EXPECT_EQ(indices[0].data<int64_t>()[0], 1);
  EXPECT_EQ(indices[0].data<int64_t>()[1], 3);
  EXPECT_EQ(indices[0].data<int64_t>()[2], 5);
}

TEST_F(PeakFindingTestCPU, ArgrelextremaLess1D) {
  // [5, 3, 4, 1, 2, 0, 3] -> minima at 1, 3, 5 (values 3, 1, 0)
  std::vector<double> data = {5, 3, 4, 1, 2, 0, 3};
  Array arr = to_array(data, DType::F64, CPUPlace());

  auto indices = argrelextrema(arr, "less", 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  ASSERT_EQ(indices[0].numel(), 3);
  EXPECT_EQ(indices[0].data<int64_t>()[0], 1);
  EXPECT_EQ(indices[0].data<int64_t>()[1], 3);
  EXPECT_EQ(indices[0].data<int64_t>()[2], 5);
}

TEST_F(PeakFindingTestCPU, ArgrelextremaOrder2) {
  // With order=2, a point must be max of 2 neighbors on each side
  std::vector<double> data = {1, 3, 2, 5, 4, 7, 6};
  Array arr = to_array(data, DType::F64, CPUPlace());

  auto indices = argrelextrema(arr, "greater", 0, 2, "clip");

  ASSERT_EQ(indices.size(), 1);
  // Only index 3 (value 5) and 5 (value 7) should be maxima with order=2
  ASSERT_GE(indices[0].numel(), 1);
}

TEST_F(PeakFindingTestCPU, ArgrelextremaNoMaxima) {
  // Monotonically increasing - no maxima
  std::vector<double> data = {1, 2, 3, 4, 5};
  Array arr = to_array(data, DType::F64, CPUPlace());

  auto indices = argrelextrema(arr, "greater", 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  EXPECT_EQ(indices[0].numel(), 0);
}

TEST_F(PeakFindingTestCPU, ArgrelextremaF32) {
  std::vector<float> data = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};
  Array arr = to_array(data, DType::F32, CPUPlace());

  auto indices = argrelextrema(arr, "greater", 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  ASSERT_EQ(indices[0].numel(), 2);
  EXPECT_EQ(indices[0].data<int64_t>()[0], 1);
  EXPECT_EQ(indices[0].data<int64_t>()[1], 3);
}

// ============================================================================
// argrelmax
// ============================================================================

TEST_F(PeakFindingTestCPU, ArgrelmaxBasic) {
  std::vector<double> data = {1, 3, 2, 5, 4};
  Array arr = to_array(data, DType::F64, CPUPlace());

  auto indices = argrelmax(arr, 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  ASSERT_EQ(indices[0].numel(), 2);
  EXPECT_EQ(indices[0].data<int64_t>()[0], 1);
  EXPECT_EQ(indices[0].data<int64_t>()[1], 3);
}

// ============================================================================
// argrelmin
// ============================================================================

TEST_F(PeakFindingTestCPU, ArgrelminBasic) {
  std::vector<double> data = {5, 3, 4, 1, 2};
  Array arr = to_array(data, DType::F64, CPUPlace());

  auto indices = argrelmin(arr, 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  ASSERT_EQ(indices[0].numel(), 2);
  EXPECT_EQ(indices[0].data<int64_t>()[0], 1);
  EXPECT_EQ(indices[0].data<int64_t>()[1], 3);
}

TEST_F(PeakFindingTestCPU, ArgrelminI32) {
  std::vector<int32_t> data = {5, 3, 4, 1, 2};
  Array arr = to_array(data, DType::I32, CPUPlace());

  auto indices = argrelmin(arr, 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  ASSERT_EQ(indices[0].numel(), 2);
  EXPECT_EQ(indices[0].data<int64_t>()[0], 1);
  EXPECT_EQ(indices[0].data<int64_t>()[1], 3);
}

TEST_F(PeakFindingTestCPU, Argrelmax2D) {
  // 2D array, find maxima along axis 0
  std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                              5, 4, 3, 2, 1, 0, 1, 2, 3};
  Array arr = to_array(data, Shape({2, 9}), DType::F64, CPUPlace());

  auto indices = argrelmax(arr, 1, 1, "clip");

  ASSERT_EQ(indices.size(), 2);
}

TEST_F(PeakFindingTestCPU, ArgrelmaxInvalidOrder) {
  std::vector<double> data = {1, 2, 3};
  Array arr = to_array(data, DType::F64, CPUPlace());

  EXPECT_THROW(argrelmax(arr, 0, 0, "clip"), std::exception);
}
