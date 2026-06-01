--- Signal I/O functions.
-- Provides binary file read/write, pack/unpack, and SigMF metadata support.
-- @module insight.signal.io

local native = require("_insight")
local sig = native.signal

local M = {}

--- Read a binary file into an array.
-- @string filepath Path to the binary file.
-- @string dtype Data type string (e.g., 'float32', 'int16').
-- @treturn Array Data read from the file.
function M.read_bin(filepath, dtype)
  return sig.read_bin(filepath, dtype)
end

--- Write an array to a binary file.
-- @tparam Array x Input array.
-- @string filepath Path to the output binary file.
function M.write_bin(x, filepath)
  return sig.write_bin(x, filepath)
end

--- Unpack binary data from raw bytes.
-- @tparam Array raw Raw byte array.
-- @string dtype Target data type string.
-- @treturn Array Unpacked data array.
function M.unpack_bin(raw, dtype)
  return sig.unpack_bin(raw, dtype)
end

--- Pack array data into raw bytes.
-- @tparam Array x Input array.
-- @string dtype Target data type string.
-- @treturn Array Packed byte array.
function M.pack_bin(x, dtype)
  return sig.pack_bin(x, dtype)
end

--- Read a SigMF metadata file and its associated data.
-- @string metafile Path to the SigMF .json metadata file.
-- @treturn Array Data array read from the associated binary file.
function M.read_sigmf(metafile)
  return sig.read_sigmf(metafile)
end

--- Write data and metadata in SigMF format.
-- @tparam Array x Input data array.
-- @string metafile Path to the SigMF .json metadata file.
-- @tparam table metadata SigMF metadata as a Lua table.
function M.write_sigmf(x, metafile, metadata)
  return sig.write_sigmf(x, metafile, metadata)
end

return M
