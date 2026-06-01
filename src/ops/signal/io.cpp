// src/ops/signal/io.cpp
//
// I/O functions are exempt from the no-raw-pointer rule because they
// interface with the operating system's file I/O API which requires
// raw buffer access (ifs.read, ofs.write, memcpy for binary data).
//
#include "insight/ops/signal/io.h"
#include "insight/c_api/dtype.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

namespace ins {
namespace signal {

// ============================================================================
// read_bin
// ============================================================================

Array read_bin(const std::string &file, DType dtype, int64_t num_samples,
               int64_t offset) {
  INS_CHECK(!file.empty(), "read_bin: file path is empty");

  std::ifstream ifs(file, std::ios::binary | std::ios::ate);
  INS_CHECK(ifs.is_open(), "read_bin: cannot open file: " + file);

  int64_t file_size = ifs.tellg();
  int64_t dsize = insight_dtype_size(static_cast<int32_t>(dtype));
  int64_t skip_bytes = offset * dsize;
  int64_t available = (file_size - skip_bytes) / dsize;

  INS_CHECK(skip_bytes <= file_size, "read_bin: offset exceeds file size");

  if (num_samples <= 0)
    num_samples = available;
  else
    num_samples = std::min(num_samples, available);

  INS_CHECK(num_samples > 0, "read_bin: no samples to read");

  ifs.seekg(skip_bytes);

  Place cpu = CPUPlace();
  Array result = zeros(Shape({num_samples}), dtype, cpu);

  // Read into array
  ifs.read(reinterpret_cast<char *>(result.data<char>()), num_samples * dsize);

  return result;
}

// ============================================================================
// write_bin
// ============================================================================

void write_bin(const std::string &file, const Array &data, bool append) {
  INS_CHECK(data.defined(), "write_bin: data is undefined");

  Place cpu = CPUPlace();
  Array data_cpu =
      (data.place().kind() == DeviceKind::CPU) ? data : data.to(cpu);

  std::ios::openmode mode = std::ios::binary;
  if (append)
    mode |= std::ios::app;

  std::ofstream ofs(file, mode);
  INS_CHECK(ofs.is_open(), "write_bin: cannot open file: " + file);

  int64_t dtype_size = insight_dtype_size(static_cast<int32_t>(data.dtype()));
  ofs.write(reinterpret_cast<const char *>(data_cpu.data<char>()),
            data_cpu.numel() * dtype_size);
  ofs.close();
}

// ============================================================================
// unpack_bin
// ============================================================================

Array unpack_bin(const Array &binary, DType dtype,
                 const std::string &endianness) {
  INS_CHECK(binary.defined(), "unpack_bin: input is undefined");

  Place cpu = CPUPlace();
  Array bin_cpu =
      (binary.place().kind() == DeviceKind::CPU) ? binary : binary.to(cpu);

  int64_t dtype_size = insight_dtype_size(static_cast<int32_t>(dtype));
  int64_t n = bin_cpu.numel() / dtype_size;

  INS_CHECK(n > 0, "unpack_bin: binary data too small for dtype");

  Array result = zeros(Shape({n}), dtype, cpu);

  const uint8_t *src = bin_cpu.data<uint8_t>();
  uint8_t *dst = result.data<uint8_t>();

  bool need_swap = (endianness == "B"); // big-endian needs byte swap on LE
                                        // systems

  for (int64_t i = 0; i < n; ++i) {
    const uint8_t *src_ptr = src + i * dtype_size;
    uint8_t *dst_ptr = dst + i * dtype_size;

    if (need_swap) {
      // Byte swap
      for (int64_t j = 0; j < dtype_size; ++j) {
        dst_ptr[j] = src_ptr[dtype_size - 1 - j];
      }
    } else {
      std::memcpy(dst_ptr, src_ptr, dtype_size);
    }
  }

  return result;
}

// ============================================================================
// pack_bin
// ============================================================================

Array pack_bin(const Array &data) {
  INS_CHECK(data.defined(), "pack_bin: input is undefined");

  Place cpu = CPUPlace();
  Array data_cpu =
      (data.place().kind() == DeviceKind::CPU) ? data : data.to(cpu);

  int64_t dtype_size = insight_dtype_size(static_cast<int32_t>(data.dtype()));
  int64_t total_bytes = data_cpu.numel() * dtype_size;

  Array result = zeros(Shape({total_bytes}), DType::U8, cpu);
  std::memcpy(result.data<uint8_t>(), data_cpu.data<char>(), total_bytes);

  return result;
}

// ============================================================================
// read_sigmf
// ============================================================================

Array read_sigmf(const std::string &data_file, const std::string &meta_file,
                 int64_t num_samples, int64_t offset) {
  INS_CHECK(!data_file.empty(), "read_sigmf: data_file is empty");

  // Derive meta file path if not specified
  std::string meta = meta_file;
  if (meta.empty()) {
    meta = data_file;
    // Replace .sigmf-data with .sigmf-meta
    auto pos = meta.find(".sigmf-data");
    if (pos != std::string::npos) {
      meta.replace(pos, 11, ".sigmf-meta");
    }
  }

  // Try to parse metadata for dtype/endianness
  DType dtype = DType::C32; // default SigMF type
  std::string endianness = "L";

  std::ifstream mfs(meta);
  if (mfs.is_open()) {
    // Simple JSON parsing for SigMF metadata
    std::string content((std::istreambuf_iterator<char>(mfs)),
                        std::istreambuf_iterator<char>());

    // Find "core:datatype" value
    auto dt_pos = content.find("\"core:datatype\"");
    if (dt_pos != std::string::npos) {
      auto colon = content.find(':', dt_pos);
      auto quote1 = content.find('"', colon + 1);
      auto quote2 = content.find('"', quote1 + 1);
      std::string dt_str = content.substr(quote1 + 1, quote2 - quote1 - 1);

      if (dt_str == "cf32_le" || dt_str == "cf32") {
        dtype = DType::C32;
        endianness = "L";
      } else if (dt_str == "cf32_be") {
        dtype = DType::C32;
        endianness = "B";
      } else if (dt_str == "cf64_le" || dt_str == "cf64") {
        dtype = DType::C64;
        endianness = "L";
      } else if (dt_str == "cf64_be") {
        dtype = DType::C64;
        endianness = "B";
      } else if (dt_str == "ri16_le" || dt_str == "ri16") {
        dtype = DType::I16;
        endianness = "L";
      } else if (dt_str == "ri16_be") {
        dtype = DType::I16;
        endianness = "B";
      } else if (dt_str == "rf32_le" || dt_str == "rf32") {
        dtype = DType::F32;
        endianness = "L";
      } else if (dt_str == "rf32_be") {
        dtype = DType::F32;
        endianness = "B";
      }
    }
  }

  // Read binary data
  Array raw = read_bin(data_file, DType::U8, 0, 0);

  // Unpack if needed
  if (endianness == "B") {
    return unpack_bin(raw, dtype, endianness);
  }

  // Reinterpret as correct dtype
  Place cpu = CPUPlace();
  int64_t dtype_size = insight_dtype_size(static_cast<int32_t>(dtype));
  int64_t n = raw.numel() / dtype_size;

  // Copy data to correctly typed array
  Array result = zeros(Shape({n}), dtype, cpu);
  std::memcpy(result.data<char>(), raw.data<char>(), raw.numel());

  // Apply offset/num_samples
  if (offset > 0 || num_samples > 0) {
    int64_t start = offset;
    int64_t count = (num_samples > 0) ? num_samples : (n - start);
    count = std::min(count, n - start);

    Array sliced = zeros(Shape({count}), dtype, cpu);
    std::memcpy(sliced.data<char>(), result.data<char>() + start * dtype_size,
                count * dtype_size);
    return sliced;
  }

  return result;
}

// ============================================================================
// write_sigmf
// ============================================================================

void write_sigmf(const std::string &data_file, const Array &data, bool append) {
  write_bin(data_file, data, append);
}

} // namespace signal
} // namespace ins
