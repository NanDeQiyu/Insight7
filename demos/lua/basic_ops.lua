-- demos/lua/basic_ops.lua
-- Demonstrates: array creation, arithmetic, multi-dtype, broadcasting,
-- unary operations, and reductions.

local ins = require("insight")

local function separator(title)
  print(string.rep("=", 40))
  print("  " .. title)
  print(string.rep("=", 40))
end

ins.init({ "cpu" })

print("Insight7 Basic Operations Demo (Lua)")

-- --- Array creation ---
separator("Array Creation")

local a = ins.ones({ 3, 4 }, ins.float32)
print(string.format("ones(3,4) shape=[%d,%d] dtype=%s", a.shape[1], a.shape[2], tostring(a.dtype)))

local b = ins.arange(12, ins.float64):reshape({ 3, 4 })
print("arange(0,12).reshape(3,4):")
print(tostring(b))

local c = ins.eye(4, -1, 0, ins.float64)
print("eye(4) F64:")
print(tostring(c))

-- --- Multi-dtype arithmetic ---
separator("Multi-Dtype Arithmetic")

-- F32
local f32_a = ins.from_table({ 1, 2, 3 }):to(ins.float32)
local f32_b = ins.from_table({ 4, 5, 6 }):to(ins.float32)
local f32_c = f32_a + f32_b
print(string.format("F32: [1,2,3] + [4,5,6] = [%g,%g,%g]", f32_c[1], f32_c[2], f32_c[3]))

-- F64
local f64_a = ins.from_table({ 1, 2, 3 })
local f64_b = ins.from_table({ 0.5, 0.5, 0.5 })
local f64_c = f64_a * f64_b
print(string.format("F64: [1,2,3] * [0.5,0.5,0.5] = [%g,%g,%g]", f64_c[1], f64_c[2], f64_c[3]))

-- I32
local i32_a = ins.from_table({ 10, 20, 30 }):to(ins.int32)
local i32_b = ins.from_table({ 3, 3, 3 }):to(ins.int32)
local i32_c = i32_a / i32_b
print(string.format("I32: [10,20,30] / [3,3,3] = [%d,%d,%d]", i32_c[1], i32_c[2], i32_c[3]))

-- --- Broadcasting ---
separator("Broadcasting")

local m = ins.from_table({ { 1, 2, 3 }, { 4, 5, 6 } })
local row = ins.from_table({ { 10, 20, 30 } })
local result = m + row
print("Matrix + Row broadcast:")
print(tostring(result))

-- --- Unary operations ---
separator("Unary Operations")

local x = ins.from_table({ -2, -1, 0, 1, 2 })
print("x:      " .. tostring(x))
print("abs(x): " .. tostring(ins.abs(x)))
print("sin(x): " .. tostring(ins.sin(x)))
print("exp(x): " .. tostring(ins.exp(x)))

-- --- Reductions ---
separator("Reductions")

local data = ins.from_table({ 1, 2, 3, 4, 5 })
print("data:  " .. tostring(data))
print("sum:   " .. tostring(ins.sum(data)))
print("mean:  " .. tostring(ins.mean(data)))
print("max:   " .. tostring(ins.max(data)))
print("min:   " .. tostring(ins.min(data)))

print("\nDone!")
