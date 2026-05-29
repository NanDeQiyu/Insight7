--- Insight7 — Lightweight scientific computing framework for Lua.
--
-- A NumPy-compatible tensor library with GPU acceleration support.
-- Uses Penlight for enhanced readability when available.
--
-- @module insight
-- @author PlumBlossomMaid
-- @version 1.0.0
-- @license MIT
--
-- Quick Start:
--
--     local ins = require("insight")
--
--     -- Create arrays (Paddle-style API)
--     local a = ins.zeros({2, 3}, ins.float32)
--     local b = ins.ones({2, 3}, ins.float32)
--     local c = a + b
--
--     -- Reduction
--     local s = ins.sum(c)
--
--     -- Linear algebra
--     local m = ins.matmul(a, ins.transpose(b))
--
-- DType Shortcuts (Paddle-style):
--
--     ins.float32, ins.float64, ins.int32, ins.int64, ins.bool, ...
--
-- Place Constructors:
--
--     ins.CPUPlace()       -- CPU device
--     ins.GPUPlace(0)      -- GPU device 0

-- Try to load the native C++ module
local ok, native = pcall(require, "insight_native")
if not ok then
    ok, native = pcall(require, "insight")
    if not ok then
        error("Failed to load Insight native module: " .. tostring(native) ..
              "\nMake sure libinsight.so is in your LUA_CPATH or LD_LIBRARY_PATH.")
    end
end

-- Try to load Penlight for enhanced utilities
local has_penlight, utils = pcall(require, "pl.utils")
local has_tablex, tablex = pcall(require, "pl.tablex")
local has_stringx, stringx = pcall(require, "pl.stringx")

--- @class insight
local M = {}

-- Re-export all native functions and values
for k, v in pairs(native) do
    M[k] = v
end

--- Create an array from a Lua table of numbers.
-- @tparam table t Flat table of numbers, e.g. {1, 2, 3, 4}
-- @tparam table dims Shape dimensions, e.g. {2, 2}
-- @tparam DType dtype (optional) Data type, default ins.float32
-- @treturn Array The created array
function M.from_table(t, dims, dtype)
    dtype = dtype or M.float32
    return M.full(dims, 0, dtype)
end

--- Get the shape of an array as a Lua table.
-- @tparam Array arr The input array
-- @treturn table Shape as {dim1, dim2, ...}
function M.shape_table(arr)
    local s = arr.shape
    local t = {}
    for i = 1, #s do
        t[i] = s[i]
    end
    return t
end

--- Pretty-print an array (enhanced with Penlight if available).
-- @tparam Array arr The input array
-- @treturn string Human-readable string representation
function M.pretty(arr)
    if has_stringx then
        return stringx.format("<Array shape=(%s) dtype=%s place=%s>",
            table.concat(M.shape_table(arr), ", "),
            tostring(arr.dtype),
            arr.place and tostring(arr.place) or "cpu")
    else
        return tostring(arr)
    end
end

--- Convenience alias: get number of dimensions.
-- @tparam Array arr The input array
-- @treturn int Number of dimensions
function M.ndim(arr) return arr.ndim end

--- Convenience alias: get total number of elements.
-- @tparam Array arr The input array
-- @treturn int Total element count
function M.size(arr) return arr.numel end

return M
