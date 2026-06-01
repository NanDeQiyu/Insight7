// tests/cpu/test_device_info.cpp
#include "insight/core/place.h"
#include "insight/init.h"
#include <gtest/gtest.h>

using namespace ins;

class DeviceInfoTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ins::init(); }
};

TEST_F(DeviceInfoTestCPU, DeviceCount) {
  size_t count = device_count(DeviceKind::CPU);
  EXPECT_GE(count, 0);
}

TEST_F(DeviceInfoTestCPU, DeviceNameCPU) {
  std::string name = device_name(DeviceKind::CPU);
  EXPECT_EQ(name, "CPU");
}

TEST_F(DeviceInfoTestCPU, CudaVersionNoGPU) {
  int ver = cuda_version();
  EXPECT_GE(ver, 0);
}

TEST_F(DeviceInfoTestCPU, DriverVersionNoGPU) {
  int ver = driver_version();
  EXPECT_GE(ver, 0);
}

TEST_F(DeviceInfoTestCPU, ComputeCapabilityNoGPU) {
  int cap = compute_capability(0);
  EXPECT_GE(cap, 0);
}

TEST_F(DeviceInfoTestCPU, DeviceMemoryNoGPU) {
  DeviceMemoryInfo info = device_memory(0);
  EXPECT_GE(info.total, static_cast<size_t>(0));
  EXPECT_GE(info.free, static_cast<size_t>(0));
}

TEST_F(DeviceInfoTestCPU, IsDeviceAvailableCPU) {
  EXPECT_TRUE(is_device_available(DeviceKind::CPU));
}
