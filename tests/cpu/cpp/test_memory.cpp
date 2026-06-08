// tests/cpu/cpp/test_memory.cpp
#include "insight/c_api/memory.h"
#include "insight/core/place.h"
#include "insight/insight.h"
#include "gtest/gtest.h"

class MemoryTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ins::init(); }
};

// ========== C++ API ==========

TEST_F(MemoryTestCPU, DeviceMemoryInfoCPU) {
  auto info = ins::device_memory_info(ins::DeviceKind::CPU);
  // CPU should report some memory (at least > 0)
  EXPECT_GT(info.total, 0);
  EXPECT_GT(info.free, 0);
}

TEST_F(MemoryTestCPU, DeviceMemoryInfoCPUSpecific) {
  auto info = ins::device_memory_info(ins::DeviceKind::CPU, 0);
  EXPECT_GT(info.total, 0);
  EXPECT_GT(info.free, 0);
}

// ========== C API ==========

TEST_F(MemoryTestCPU, CDeviceMemoryInfoCPU) {
  size_t total = 0, free_mem = 0;
  C_Status status = insight_device_memory_info(0, 0, &total, &free_mem);
  EXPECT_EQ(status, C_SUCCESS);
  EXPECT_GT(total, 0);
  EXPECT_GT(free_mem, 0);
}

TEST_F(MemoryTestCPU, CDeviceMemoryInfoNullPtrs) {
  EXPECT_NE(insight_device_memory_info(0, 0, nullptr, nullptr), C_SUCCESS);
}
