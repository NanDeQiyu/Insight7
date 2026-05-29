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

-- Try to load the native C++ module (_insight.so)
local ok, native = pcall(require, "_insight")
if not ok then
    error("Failed to load Insight native module: " .. tostring(native) ..
          "\nMake sure _insight.so is in your LUA_CPATH or LD_LIBRARY_PATH.")
end

-- Try to load Penlight for enhanced utilities
local has_penlight, utils = pcall(require, "pl.utils")
local has_tablex, tablex = pcall(require, "pl.tablex")
local has_stringx, stringx = pcall(require, "pl.stringx")

--- @class insight
local M = {}

-- Save native reference for internal use
M._native = native

-- Re-export all native functions and values
for k, v in pairs(native) do
    M[k] = v
end

--- Create an Array from a nested Lua table.
-- Strict validation: only number, boolean, and table allowed.
-- nil, string, function, userdata → error. Max 10 dimensions.
-- @tparam table t Nested table of numbers/booleans
-- @treturn Array The created array (float64)
function M.from_table(t)
    local function validate(obj, path, depth)
        if depth > 10 then
            error("Table nesting exceeds 10 dimensions at " .. path, 3)
        end
        local tp = type(obj)
        if tp == "nil" then
            error("nil found at " .. path, 3)
        elseif tp == "string" then
            error("string found at " .. path, 3)
        elseif tp == "function" then
            error("function found at " .. path, 3)
        elseif tp == "userdata" or tp == "lightuserdata" then
            error("userdata found at " .. path, 3)
        elseif tp == "thread" then
            error("thread found at " .. path, 3)
        elseif tp ~= "number" and tp ~= "boolean" and tp ~= "table" then
            error("unexpected type '" .. tp .. "' at " .. path, 3)
        end
    end

    local function infer_shape(tbl, shape, depth, path)
        validate(tbl, path, depth)
        if type(tbl) ~= "table" then return end
        local len = #tbl
        if depth > #shape then
            shape[#shape + 1] = len
        elseif shape[depth] ~= len then
            error("Ragged table at " .. path .. ": dimension " .. depth ..
                  " inconsistent (" .. shape[depth] .. " vs " .. len .. ")", 3)
        end
        for i = 1, len do
            validate(tbl[i], path .. "[" .. i .. "]", depth)
            if type(tbl[i]) == "table" then
                infer_shape(tbl[i], shape, depth + 1, path .. "[" .. i .. "]")
            end
        end
    end

    if type(t) ~= "table" then
        error("from_table expects a table, got " .. type(t), 2)
    end
    if #t == 0 then
        return M.zeros({0}, M.float64)
    end
    infer_shape(t, {}, 1, "root")
    return M._native.from_table(t)
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
