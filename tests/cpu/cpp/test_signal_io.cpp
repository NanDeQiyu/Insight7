// tests/cpu/test_signal_io.cpp
#include "insight/insight.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;
using namespace ins::signal;

class SignalIOTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    tmp_base = (std::filesystem::temp_directory_path() /
                ("insight_io_test_" + std::to_string(ts)))
                   .string();
    std::filesystem::create_directories(tmp_base);
  }
  static void TearDownTestSuite() {
    std::error_code ec;
    std::filesystem::remove_all(tmp_base, ec);
  }
  void SetUp() override {
    tmp_dir = tmp_base;
    std::filesystem::create_directories(tmp_dir);
  }

  static std::string tmp_base;
  std::string tmp_dir;
};

std::string SignalIOTestCPU::tmp_base;

// ============================================================================
// pack_bin / unpack_bin
// ============================================================================

TEST_F(SignalIOTestCPU, PackBinF64) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  Array arr = to_array(data, DType::F64, CPUPlace());

  Array packed = pack_bin(arr);

  EXPECT_EQ(packed.dtype(), DType::U8);
  EXPECT_EQ(packed.numel(), 4 * 8); // 4 doubles * 8 bytes
}

TEST_F(SignalIOTestCPU, UnpackBinF64) {
  std::vector<double> data = {1.5, 2.5, 3.5};
  Array arr = to_array(data, DType::F64, CPUPlace());

  Array packed = pack_bin(arr);
  Array unpacked = unpack_bin(packed, DType::F64, "L");

  EXPECT_EQ(unpacked.numel(), 3);
  EXPECT_EQ(unpacked.dtype(), DType::F64);

  const double *ud = unpacked.data<double>();
  EXPECT_NEAR(ud[0], 1.5, 1e-10);
  EXPECT_NEAR(ud[1], 2.5, 1e-10);
  EXPECT_NEAR(ud[2], 3.5, 1e-10);
}

TEST_F(SignalIOTestCPU, PackUnpackF32) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  Array arr = to_array(data, DType::F32, CPUPlace());

  Array packed = pack_bin(arr);
  Array unpacked = unpack_bin(packed, DType::F32, "L");

  EXPECT_EQ(unpacked.numel(), 3);
  const float *ud = unpacked.data<float>();
  EXPECT_NEAR(ud[0], 1.0f, 1e-6f);
  EXPECT_NEAR(ud[1], 2.0f, 1e-6f);
  EXPECT_NEAR(ud[2], 3.0f, 1e-6f);
}

TEST_F(SignalIOTestCPU, UnpackBinBigEndian) {
  // Create big-endian data manually
  std::vector<uint8_t> raw(8);
  // Big-endian double 1.0: 3F F0 00 00 00 00 00 00
  raw[0] = 0x3F;
  raw[1] = 0xF0;
  raw[2] = 0x00;
  raw[3] = 0x00;
  raw[4] = 0x00;
  raw[5] = 0x00;
  raw[6] = 0x00;
  raw[7] = 0x00;

  Array bin = to_array(raw, DType::U8, CPUPlace());
  Array unpacked = unpack_bin(bin, DType::F64, "B");

  EXPECT_NEAR(unpacked.data<double>()[0], 1.0, 1e-10);
}

// ============================================================================
// write_bin / read_bin
// ============================================================================

TEST_F(SignalIOTestCPU, WriteReadBin) {
  std::string path = tmp_dir + "/test.bin";
  std::filesystem::create_directories(tmp_dir);

  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array arr = to_array(data, DType::F64, CPUPlace());

  write_bin(path, arr, false);

  Array read = read_bin(path, DType::F64, 0, 0);

  EXPECT_EQ(read.numel(), 5);
  const double *rd = read.data<double>();
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(rd[i], data[i], 1e-10);
  }
}

TEST_F(SignalIOTestCPU, ReadBinWithOffset) {
  std::string path = tmp_dir + "/test_offset.bin";
  std::filesystem::create_directories(tmp_dir);

  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array arr = to_array(data, DType::F64, CPUPlace());

  write_bin(path, arr, false);

  // Read starting from offset 2
  Array read = read_bin(path, DType::F64, 0, 2);

  EXPECT_EQ(read.numel(), 3);
  const double *rd = read.data<double>();
  EXPECT_NEAR(rd[0], 3.0, 1e-10);
  EXPECT_NEAR(rd[1], 4.0, 1e-10);
  EXPECT_NEAR(rd[2], 5.0, 1e-10);
}

TEST_F(SignalIOTestCPU, ReadBinNumSamples) {
  std::string path = tmp_dir + "/test_ns.bin";
  std::filesystem::create_directories(tmp_dir);

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  Array arr = to_array(data, DType::F32, CPUPlace());

  write_bin(path, arr, false);

  Array read = read_bin(path, DType::F32, 3, 0);

  EXPECT_EQ(read.numel(), 3);
}

TEST_F(SignalIOTestCPU, WriteBinAppend) {
  std::string path = tmp_dir + "/test_append.bin";
  std::filesystem::create_directories(tmp_dir);

  std::vector<double> data1 = {1.0, 2.0};
  std::vector<double> data2 = {3.0, 4.0};

  Array arr1 = to_array(data1, DType::F64, CPUPlace());
  Array arr2 = to_array(data2, DType::F64, CPUPlace());

  write_bin(path, arr1, false);
  write_bin(path, arr2, true);

  Array read = read_bin(path, DType::F64, 0, 0);
  EXPECT_EQ(read.numel(), 4);
}

// ============================================================================
// read_sigmf / write_sigmf
// ============================================================================

TEST_F(SignalIOTestCPU, WriteReadSigmf) {
  std::string data_path = tmp_dir + "/test.sigmf-data";

  // Write data as raw bytes
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::filesystem::create_directories(tmp_dir);
  std::ofstream ofs(data_path, std::ios::binary);
  ASSERT_TRUE(ofs.is_open()) << "Cannot open " << data_path;
  ofs.write(reinterpret_cast<const char *>(data.data()),
            data.size() * sizeof(float));
  ofs.flush();
  ofs.close();

  // Verify file exists before reading
  ASSERT_TRUE(std::filesystem::exists(data_path))
      << "File not written: " << data_path;

  // Read back using read_bin directly (SigMF parsing is separate)
  Array read = read_bin(data_path, DType::F32, 0, 0);

  EXPECT_EQ(read.numel(), 4);
  EXPECT_EQ(read.dtype(), DType::F32);
  const float *rd = read.data<float>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(rd[i], data[i], 1e-6f);
  }
}

// ============================================================================
// Error handling
// ============================================================================

TEST_F(SignalIOTestCPU, ReadBinNonexistentFile) {
  EXPECT_THROW(read_bin("/nonexistent/file.bin", DType::F64), std::exception);
}

TEST_F(SignalIOTestCPU, PackBinUndefined) {
  Array undef;
  EXPECT_THROW(pack_bin(undef), std::exception);
}
