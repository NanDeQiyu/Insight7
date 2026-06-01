-- demos/lua/linalg_demo.lua
-- Demonstrates: linear algebra on CPU and GPU.

local ins = require("insight")

local function separator(title)
  print(string.rep("=", 40))
  print("  " .. title)
  print(string.rep("=", 40))
end

local function gpu_available()
  local ok, _ = pcall(function()
    ins.load_backend("cuda")
  end)
  return ok
end

local function run_cpu_linalg()
  separator("CPU Linear Algebra")

  -- MatMul F64
  local A = ins.from_table({ { 1, 2 }, { 3, 4 } })
  local B = ins.from_table({ { 5, 6 }, { 7, 8 } })
  local C = ins.matmul(A, B)
  print("MatMul F64:")
  print(tostring(C))

  -- MatMul F32
  local A32 = A:to(ins.float32)
  local B32 = B:to(ins.float32)
  local C32 = ins.matmul(A32, B32)
  print("MatMul F32:")
  print(tostring(C32))

  -- Determinant
  local ok_det, det_val = pcall(function()
    return ins.det(A)[1]
  end)
  if ok_det then
    print(string.format("det([[1,2],[3,4]]) = %g", det_val))
  else
    print("det: skipped (requires OpenBLAS)")
  end

  -- Inverse
  local ok_inv, A_inv = pcall(ins.inv, A)
  if ok_inv then
    print("inv([[1,2],[3,4]]):")
    print(tostring(A_inv))
    local I = ins.matmul(A, A_inv)
    print("A * A_inv (should be identity):")
    print(tostring(I))
  else
    print("inv: skipped (requires OpenBLAS)")
  end

  -- SVD
  local D = ins.from_table({ { 1, 0, 0 }, { 0, 2, 0 }, { 0, 0, 3 } })
  local ok_svd, U, S, VT = pcall(ins.svd, D, false)
  if ok_svd then
    print("SVD singular values: " .. tostring(S))
  else
    print("SVD: skipped (requires OpenBLAS)")
  end

  -- Solve linear system
  local A3 = ins.from_table({ { 3, 2, -1 }, { 2, -2, 4 }, { -1, 0.5, -1 } })
  local b = ins.from_table({ 1, -2, 0 })
  local ok_solve, x = pcall(ins.solve, A3, b)
  if ok_solve then
    print("Ax=b solution: " .. tostring(x))
  else
    print("solve: skipped (requires OpenBLAS)")
  end
end

local function run_gpu_linalg()
  separator("GPU Linear Algebra")

  local A = ins.from_table({ { 1, 2 }, { 3, 4 } }):to(ins.GPUPlace(0))
  local B = ins.from_table({ { 5, 6 }, { 7, 8 } }):to(ins.GPUPlace(0))
  local C = ins.matmul(A, B)
  print("GPU MatMul F64:")
  print(tostring(C:to(ins.CPUPlace())))

  local A32 = ins.from_table({ { 1, 2 }, { 3, 4 } }):to(ins.float32):to(ins.GPUPlace(0))
  local B32 = ins.from_table({ { 5, 6 }, { 7, 8 } }):to(ins.float32):to(ins.GPUPlace(0))
  local C32 = ins.matmul(A32, B32)
  print("GPU MatMul F32:")
  print(tostring(C32:to(ins.CPUPlace())))

  local A = ins.from_table({ { 1, 2 }, { 3, 4 } }):to(ins.GPUPlace(0))
  local ok_gdet, gdet = pcall(function()
    return ins.det(A):to(ins.CPUPlace())[1]
  end)
  if ok_gdet then
    print(string.format("GPU det = %g", gdet))
  else
    print("GPU det: skipped (requires OpenBLAS)")
  end
  local ok_ginv, A_inv_gpu = pcall(function()
    return ins.inv(A):to(ins.CPUPlace())
  end)
  if ok_ginv then
    print("GPU inv:")
    print(tostring(A_inv_gpu))
  else
    print("GPU inv: skipped (requires OpenBLAS)")
  end

  local D = ins.from_table({ { 1, 0, 0 }, { 0, 2, 0 }, { 0, 0, 3 } }):to(ins.float32):to(ins.GPUPlace(0))
  local ok_gsvd, U, S, VT = pcall(ins.svd, D, false)
  if ok_gsvd then
    print("GPU SVD singular values (F32): " .. tostring(S:to(ins.CPUPlace())))
  else
    print("GPU SVD: skipped (requires OpenBLAS)")
  end
end

local ok = pcall(ins.init, { "cpu", "cuda" })
if not ok then
  ins.init({ "cpu" })
end

print("Insight7 Linear Algebra Demo (Lua)")

run_cpu_linalg()

if gpu_available() then
  run_gpu_linalg()
else
  print("\n[GPU not available, skipping GPU linalg demo]")
end

print("\nDone!")
