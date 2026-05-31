// insight/ops/signal/io.h
#pragma once
#include "insight/core/array.h"
#include <cstdint>
#include <string>

namespace ins {
namespace signal {

/**
 * @brief Read binary file into an array.
 *
 * @param file Path to binary file
 * @param dtype Data type for interpretation. Default: U8
 * @param num_samples Number of samples to read. 0 = all. Default: 0
 * @param offset Byte offset into file. Default: 0
 * @return Array with data
 */
Array read_bin(const std::string &file, DType dtype = DType::U8,
               int64_t num_samples = 0, int64_t offset = 0);

/**
 * @brief Write array to binary file.
 *
 * @param file Output file path
 * @param data Data array to write
 * @param append Append mode. Default: true
 */
void write_bin(const std::string &file, const Array &data, bool append = true);

/**
 * @brief Unpack binary data to specified dtype with endianness conversion.
 *
 * @param binary Input binary array (U8)
 * @param dtype Target data type
 * @param endianness "L" (little-endian) or "B" (big-endian). Default: "L"
 * @return Unpacked array
 */
Array unpack_bin(const Array &binary, DType dtype,
                 const std::string &endianness = "L");

/**
 * @brief Pack array to raw bytes (U8).
 *
 * @param data Input array
 * @return Byte array (U8)
 */
Array pack_bin(const Array &data);

/**
 * @brief Read SigMF format binary data.
 *
 * Parses .sigmf-meta JSON for datatype/endianness and reads/parses
 * the corresponding .sigmf-data file.
 *
 * @param data_file Path to .sigmf-data file
 * @param meta_file Path to .sigmf-meta file. Default: derived from data_file
 * @param num_samples Number of samples to read. 0 = all. Default: 0
 * @param offset Sample offset. Default: 0
 * @return Array with parsed data
 */
Array read_sigmf(const std::string &data_file,
                 const std::string &meta_file = "", int64_t num_samples = 0,
                 int64_t offset = 0);

/**
 * @brief Write array in SigMF format.
 *
 * Packs data and writes to .sigmf-data file.
 *
 * @param data_file Output .sigmf-data path
 * @param data Data array to write
 * @param append Append mode. Default: true
 */
void write_sigmf(const std::string &data_file, const Array &data,
                 bool append = true);

} // namespace signal
} // namespace ins
