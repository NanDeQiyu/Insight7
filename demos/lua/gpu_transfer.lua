-- demos/lua/gpu_transfer.lua
-- Demonstrates: GPU device management, CPU-GPU data transfer.

local ins = require("insight")

local function separator(title)
  print(string.rep("=", 40))
  print("  " .. title)
  print(string.rep("=", 40))
end

local function gpu_available()
  local ok, result = pcall(function()
    return ins.load_backend("cuda")
  end)
  return ok and result
end

ins.init()

print("Insight7 GPU Transfer Demo (Lua)")

if not gpu_available() then
  print("GPU not available. This demo requires a CUDA device.")
  os.exit(0)
end

-- --- CPU to GPU transfer ---
separator("CPU -> GPU Transfer")

local cpu_arr = ins.from_table({ { 1, 2, 3 }, { 4, 5, 6 } })
print("CPU array:")
print(tostring(cpu_arr))

local gpu_arr = cpu_arr:to(ins.GPUPlace(0))
print(
  string.format(
    "GPU array shape=[%d,%d] place=%s",
    gpu_arr.shape[1],
    gpu_arr.shape[2],
    gpu_arr.place.is_gpu and "gpu" or "cpu"
  )
)

-- --- GPU to CPU transfer ---
separator("GPU -> CPU Transfer")

local back = gpu_arr:to(ins.CPUPlace())
print("Round-tripped back to CPU:")
print(tostring(back))

-- --- GPU arithmetic ---
separator("GPU Arithmetic (F64)")

local a = ins.from_table({ 1, 2, 3 }):to(ins.GPUPlace(0))
local b = ins.from_table({ 4, 5, 6 }):to(ins.GPUPlace(0))

local sum_ab = a + b
local mul_ab = a * b
print("GPU [1,2,3] + [4,5,6] = " .. tostring(sum_ab:to(ins.CPUPlace())))
print("GPU [1,2,3] * [4,5,6] = " .. tostring(mul_ab:to(ins.CPUPlace())))

-- --- GPU arithmetic F32 ---
separator("GPU Arithmetic (F32)")

local a32 = ins.from_table({ 1, 2, 3 }):to(ins.float32):to(ins.GPUPlace(0))
local b32 = ins.from_table({ 4, 5, 6 }):to(ins.float32):to(ins.GPUPlace(0))
local c32 = a32 * b32 + a32
print("GPU F32: [1,2,3]*[4,5,6]+[1,2,3] = " .. tostring(c32:to(ins.CPUPlace())))

-- --- GPU reductions ---
separator("GPU Reductions")

local data = ins.from_table({ 1, 2, 3, 4, 5 }):to(ins.GPUPlace(0))
print("GPU sum:  " .. tostring(ins.sum(data):to(ins.CPUPlace())))
print("GPU mean: " .. tostring(ins.mean(data):to(ins.CPUPlace())))
print("GPU max:  " .. tostring(ins.max(data):to(ins.CPUPlace())))

-- --- Zero-copy views ---
separator("Zero-Copy Views (GPU)")

local base = ins.arange(12, ins.float32):reshape({ 3, 4 })
local gpu_base = base:to(ins.GPUPlace(0))

-- Reshape
local reshaped = gpu_base:reshape({ 4, 3 })
print(string.format("Reshape 3x4 -> 4x3: shape=[%d,%d]", reshaped.shape[1], reshaped.shape[2]))

-- Transpose
local transposed = gpu_base:transpose()
print(string.format("Transpose 3x4 -> 4x3: shape=[%d,%d]", transposed.shape[1], transposed.shape[2]))

-- Slice
local sliced = gpu_base["1:2"]
print(string.format("Slice rows 1-2: shape=[%d,%d]", sliced.shape[1], sliced.shape[2]))
print("Sliced data: " .. tostring(sliced:to(ins.CPUPlace())))

print("\nDone!")
