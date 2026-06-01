# CPU basic operations tests — aligned with C++ test suite
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_cpu_basic.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

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
println("=== Creation ===")

a = Insight.zeros([2, 3], Insight.float32)
check("zeros numel", Insight.numel(a) == 6)

a = Insight.ones([2, 3], Insight.float32)
check("ones numel", Insight.numel(a) == 6)

a = Insight.full([2, 3], 3.14, Insight.float32)
check("full numel", Insight.numel(a) == 6)

a = Insight.eye(3)
check("eye numel", Insight.numel(a) == 9)

a = Insight.arange(10, Insight.float32)
check("arange numel", Insight.numel(a) == 10)

a = Insight.linspace(0.0, 1.0, 5, Insight.float64)
check("linspace numel", Insight.numel(a) == 5)

# ============================================================================
# Elementwise
# ============================================================================
println("=== Elementwise ===")

a = Insight.from_data([1.0, 2.0, 3.0])
b = Insight.from_data([4.0, 5.0, 6.0])
c = Insight.add(a, b)
check("add numel", Insight.numel(c) == 3)

c = Insight.sub(a, b)
check("sub numel", Insight.numel(c) == 3)

c = Insight.mul(a, b)
check("mul numel", Insight.numel(c) == 3)

c = Insight.div(a, b)
check("div numel", Insight.numel(c) == 3)

# ============================================================================
# Unary
# ============================================================================
println("=== Unary ===")

a = Insight.from_data([-3.0, -1.0, 0.0, 2.0, 4.0])
b = Insight.abs(a)
check("abs numel", Insight.numel(b) == 5)

a = Insight.from_data([1.0, 4.0, 9.0, 16.0])
b = Insight.sqrt(a)
check("sqrt numel", Insight.numel(b) == 4)

a = Insight.from_data([0.0, 1.0, 2.0])
b = Insight.exp(a)
check("exp numel", Insight.numel(b) == 3)

a = Insight.from_data([0.0, 0.5, 1.0])
b = Insight.sin(a)
check("sin numel", Insight.numel(b) == 3)

# ============================================================================
# Reduction
# ============================================================================
println("=== Reduction ===")

a = Insight.from_data([1.0, 2.0, 3.0, 4.0])
s = Insight.sum(a)
check("sum defined", Insight.numel(s) == 1)

m = Insight.mean(a)
check("mean defined", Insight.numel(m) == 1)

# ============================================================================
# Linalg
# ============================================================================
println("=== Linalg ===")

a = Insight.from_data([1.0, 2.0, 3.0, 4.0])
a = Insight.reshape(a, [2, 2])
b = Insight.from_data([5.0, 6.0, 7.0, 8.0])
b = Insight.reshape(b, [2, 2])
c = Insight.matmul(a, b)
check("matmul numel", Insight.numel(c) == 4)

d = Insight.det(a)
check("det defined", Insight.numel(d) == 1)

# ============================================================================
# FFT
# ============================================================================
println("=== FFT ===")

x = Insight.from_data([1.0, 2.0, 3.0, 4.0])
y = Insight.fft(x)
check("fft numel", Insight.numel(y) == 4)

z = Insight.ifft(y)
check("ifft numel", Insight.numel(z) == 4)

# ============================================================================
# Signal
# ============================================================================
println("=== Signal ===")

w = Insight.signal.hann(16)
check("hann numel", Insight.numel(w) == 16)

w = Insight.signal.hamming(16)
check("hamming numel", Insight.numel(w) == 16)

w = Insight.signal.blackman(32)
check("blackman numel", Insight.numel(w) == 32)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
