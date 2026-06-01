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
    local A = ins.from_table({{1, 2}, {3, 4}})
    local B = ins.from_table({{5, 6}, {7, 8}})
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
    print(string.format("det([[1,2],[3,4]]) = %g", ins.det(A)[1]))

    -- Inverse
    local A_inv = ins.inv(A)
    print("inv([[1,2],[3,4]]):")
    print(tostring(A_inv))
    local I = ins.matmul(A, A_inv)
    print("A * A_inv (should be identity):")
    print(tostring(I))

    -- SVD
    local D = ins.from_table({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}})
    local U, S, VT = ins.svd(D, false)
    print("SVD singular values: " .. tostring(S))

    -- Solve linear system
    local A3 = ins.from_table({{3, 2, -1}, {2, -2, 4}, {-1, 0.5, -1}})
    local b = ins.from_table({1, -2, 0})
    local x = ins.solve(A3, b)
    print("Ax=b solution: " .. tostring(x))
end

local function run_gpu_linalg()
    separator("GPU Linear Algebra")

    local A = ins.from_table({{1, 2}, {3, 4}}):to(ins.GPUPlace(0))
    local B = ins.from_table({{5, 6}, {7, 8}}):to(ins.GPUPlace(0))
    local C = ins.matmul(A, B)
    print("GPU MatMul F64:")
    print(tostring(C:to(ins.CPUPlace())))

    local A32 = ins.from_table({{1, 2}, {3, 4}}):to(ins.float32):to(ins.GPUPlace(0))
    local B32 = ins.from_table({{5, 6}, {7, 8}}):to(ins.float32):to(ins.GPUPlace(0))
    local C32 = ins.matmul(A32, B32)
    print("GPU MatMul F32:")
    print(tostring(C32:to(ins.CPUPlace())))

    local A = ins.from_table({{1, 2}, {3, 4}}):to(ins.GPUPlace(0))
    print(string.format("GPU det = %g", ins.det(A):to(ins.CPUPlace())[1]))
    local A_inv = ins.inv(A):to(ins.CPUPlace())
    print("GPU inv:")
    print(tostring(A_inv))

    local D = ins.from_table({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}):to(ins.float32):to(ins.GPUPlace(0))
    local U, S, VT = ins.svd(D, false)
    print("GPU SVD singular values (F32): " .. tostring(S:to(ins.CPUPlace())))
end

ins.init({"cpu", "cuda"})

print("Insight7 Linear Algebra Demo (Lua)")

run_cpu_linalg()

if gpu_available() then
    run_gpu_linalg()
else
    print("\n[GPU not available, skipping GPU linalg demo]")
end

print("\nDone!")
