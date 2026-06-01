# CUDA basic operations tests — aligned with C++ test suite
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_cuda_basic.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

# Try to load CUDA backend
try
    Insight.load_backend("cuda")
catch e
    println("CUDA backend not available: $e")
    println("Skipping GPU tests")
    exit(0)
end

passed = 0
failed = 0

function check(name, cond)
    global passed, failed
    if cond
        passed += 1
    else
        failed += 1
        println("FAIL: $name")
    end
end

# ============================================================================
# Creation
# ============================================================================
println("=== Creation (GPU) ===")

a = Insight.zeros([2, 3], Insight.float32)
check("zeros numel", Insight.numel(a) == 6)

a = Insight.ones([2, 3], Insight.float32)
check("ones numel", Insight.numel(a) == 6)

a = Insight.eye(3)
check("eye numel", Insight.numel(a) == 9)

# ============================================================================
# Elementwise
# ============================================================================
println("=== Elementwise (GPU) ===")

a = Insight.from_data([1.0, 2.0, 3.0])
b = Insight.from_data([4.0, 5.0, 6.0])
c = Insight.add(a, b)
check("add numel", Insight.numel(c) == 3)

c = Insight.mul(a, b)
check("mul numel", Insight.numel(c) == 3)

# ============================================================================
# Unary
# ============================================================================
println("=== Unary (GPU) ===")

a = Insight.from_data([-3.0, -1.0, 0.0, 2.0, 4.0])
b = Insight.abs(a)
check("abs numel", Insight.numel(b) == 5)

a = Insight.from_data([1.0, 4.0, 9.0, 16.0])
b = Insight.sqrt(a)
check("sqrt numel", Insight.numel(b) == 4)

# ============================================================================
# Reduction
# ============================================================================
println("=== Reduction (GPU) ===")

a = Insight.from_data([1.0, 2.0, 3.0, 4.0])
s = Insight.sum(a)
check("sum defined", Insight.numel(s) == 1)

# ============================================================================
# Signal
# ============================================================================
println("=== Signal (GPU) ===")

w = Insight.signal.hann(16)
check("hann numel", Insight.numel(w) == 16)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
