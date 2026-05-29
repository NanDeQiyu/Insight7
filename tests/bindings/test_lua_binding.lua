-- tests/bindings/test_lua_binding.lua
--
-- Smoke tests for the Insight7 Lua binding.
--
-- Run with:
--   cd build/bindings/lua && luajit ../../tests/bindings/test_lua_binding.lua
--   # or
--   cd build && luajit -e "package.cpath='./bindings/lua/?.so;'..package.cpath; local ins = require('insight'); print(ins.float32)"

local ins = require("insight")

local passed = 0
local failed = 0

local function assert_eq(name, actual, expected)
    if actual == expected then
        passed = passed + 1
    else
        failed = failed + 1
        print(string.format("FAIL: %s — expected %s, got %s", name, tostring(expected), tostring(actual)))
    end
end

local function assert_true(name, cond)
    if cond then
        passed = passed + 1
    else
        failed = failed + 1
        print(string.format("FAIL: %s", name))
    end
end

-- ===== DType shortcuts =====
print("=== DType ===")
assert_true("float32 exists", ins.float32 ~= nil)
assert_true("float64 exists", ins.float64 ~= nil)
assert_true("int32 exists",   ins.int32 ~= nil)
assert_true("int64 exists",   ins.int64 ~= nil)

-- ===== Place =====
print("=== Place ===")
local cpu = ins.CPUPlace()
assert_true("CPUPlace is_cpu", cpu:is_cpu())
assert_true("CPUPlace not is_gpu", not cpu:is_gpu())

-- ===== Array creation =====
print("=== Creation ===")
local a = ins.zeros({2, 3}, ins.float32)
assert_true("zeros defined", a.defined)
assert_eq("zeros numel", a.numel, 6)

local b = ins.ones({4}, ins.float64)
assert_true("ones defined", b.defined)
assert_eq("ones numel", b.numel, 4)

local c = ins.full({2, 2}, 3.14, ins.float32)
assert_true("full defined", c.defined)

local d = ins.eye(3)
assert_true("eye defined", d.defined)
assert_eq("eye numel", d.numel, 9)

local e = ins.arange(10)
assert_true("arange defined", e.defined)
assert_eq("arange numel", e.numel, 10)

-- ===== Array properties =====
print("=== Properties ===")
local f = ins.zeros({2, 3, 4})
assert_eq("ndim", f.ndim, 3)
assert_eq("numel", f.numel, 24)

-- ===== Operators =====
print("=== Operators ===")
local g = ins.ones({3}, ins.float32)
local h = ins.ones({3}, ins.float32)
local i = g + h
assert_true("add defined", i.defined)
assert_eq("add numel", i.numel, 3)

local j = g - h
assert_true("sub defined", j.defined)

local k = g * h
assert_true("mul defined", k.defined)

-- ===== Elementwise functions =====
print("=== Elementwise ===")
local l = ins.ones({3}, ins.float32)
local m = ins.exp(l)
assert_true("exp defined", m.defined)

local n = ins.sin(l)
assert_true("sin defined", n.defined)

-- ===== Reduction =====
print("=== Reduction ===")
local o = ins.ones({3, 4}, ins.float32)
local p = ins.sum(o)
assert_true("sum defined", p.defined)
assert_eq("sum numel", p.numel, 1)

-- ===== Linear Algebra =====
print("=== Linalg ===")
local q = ins.ones({2, 3}, ins.float32)
local r = ins.ones({3, 4}, ins.float32)
local s = ins.matmul(q, r)
assert_true("matmul defined", s.defined)
assert_eq("matmul numel", s.numel, 8)

local t = ins.eye(3)
local u = ins.det(t)
assert_true("det defined", u.defined)

-- ===== FFT =====
print("=== FFT ===")
local v = ins.ones({8}, ins.float64)
local w = ins.fft(v)
assert_true("fft defined", w.defined)

-- ===== Summary =====
print("")
print(string.format("Results: %d passed, %d failed", passed, failed))
if failed > 0 then
    os.exit(1)
end
