--- Signal I/O functions.
-- Provides binary file read/write, pack/unpack, and SigMF metadata support.
-- @module insight.signal.io

local native = require("_insight")
local sig = native.signal

local M = {}

local function _wrap(names, fn)
  return function(...)
    if select("#", ...) == 1 and type(select(1, ...)) == "table" then
      local t = select(1, ...)
      local has_names = false
      for k, _ in pairs(t) do
        if type(k) ~= "number" then
          has_names = true
          break
        end
      end
      if has_names then
        local pos = {}
        for i, name in ipairs(names) do
          pos[i] = t[name]
          if pos[i] == nil then
            pos[i] = t[i]
          end
        end
        return fn(table.unpack(pos, 1, #names))
      end
    end
    return fn(...)
  end
end

--- Read a binary file into an array.
-- @string filepath Path to the binary file.
-- @string dtype Data type string (e.g., 'float32', 'int16').
-- @treturn Array Data read from the file.
M.read_bin = _wrap({ "filepath", "dtype" }, function(filepath, dtype)
  return sig.read_bin(filepath, dtype)
end)

--- Write an array to a binary file.
-- @tparam Array x Input array.
-- @string filepath Path to the output binary file.
M.write_bin = _wrap({ "x", "filepath" }, function(x, filepath)
  return sig.write_bin(x, filepath)
end)

--- Unpack binary data from raw bytes.
-- @tparam Array raw Raw byte array.
-- @string dtype Target data type string.
-- @treturn Array Unpacked data array.
M.unpack_bin = _wrap({ "raw", "dtype" }, function(raw, dtype)
  return sig.unpack_bin(raw, dtype)
end)

--- Pack array data into raw bytes.
-- @tparam Array x Input array.
-- @string dtype Target data type string.
-- @treturn Array Packed byte array.
M.pack_bin = _wrap({ "x", "dtype" }, function(x, dtype)
  return sig.pack_bin(x, dtype)
end)

--- Read a SigMF metadata file and its associated data.
-- @string metafile Path to the SigMF .json metadata file.
-- @treturn Array Data array read from the associated binary file.
M.read_sigmf = _wrap({ "metafile" }, function(metafile)
  return sig.read_sigmf(metafile)
end)

--- Write data and metadata in SigMF format.
-- @tparam Array x Input data array.
-- @string metafile Path to the SigMF .json metadata file.
-- @tparam table metadata SigMF metadata as a Lua table.
M.write_sigmf = _wrap({ "x", "metafile", "metadata" }, function(x, metafile, metadata)
  return sig.write_sigmf(x, metafile, metadata)
end)

return M
