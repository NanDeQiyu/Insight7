# tests/bindings/test_binding.jl
#
# Smoke tests for the Insight7 Julia binding.
#
# Run with:
#   cd build && julia ../tests/bindings/test_binding.jl

# Load the Insight module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "build", "bindings", "julia"))

using Test

# Load the Insight module from build directory
const _insight_dir = joinpath(@__DIR__, "..", "..", "build", "bindings", "julia")
include(joinpath(_insight_dir, "Insight.jl"))

using .Insight

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

# ===== DType shortcuts =====
println("=== DType ===")
check("float32 exists", Insight.float32 isa Int32)
check("float64 exists", Insight.float64 isa Int32)
check("int32 exists", Insight.int32 isa Int32)
check("int64 exists", Insight.int64 isa Int32)

# ===== Array creation =====
println("=== Creation ===")
a = Insight.zeros(Int64[2, 3], Insight.float32)
check("zeros defined", a.ptr != C_NULL)
check("zeros numel", Insight.numel(a) == 6)

b = Insight.ones(Int64[4], Insight.float32)
check("ones defined", b.ptr != C_NULL)
check("ones numel", Insight.numel(b) == 4)

c = Insight.full(Int64[2, 2], 3.14, Insight.float32)
check("full defined", c.ptr != C_NULL)

d = Insight.eye(Int64(3), Int64(3), Insight.float32)
check("eye defined", d.ptr != C_NULL)
check("eye numel", Insight.numel(d) == 9)

e = Insight.arange(0.0, 10.0, 1.0, Insight.int64)
check("arange defined", e.ptr != C_NULL)
check("arange numel", Insight.numel(e) == 10)

# ===== Array properties =====
println("=== Properties ===")
f = Insight.zeros(Int64[2, 3, 4])
check("ndim", Insight.ndim(f) == 3)
check("numel", Insight.numel(f) == 24)
check("shape", Insight.shape(f) == [2, 3, 4])

# ===== Operators =====
println("=== Operators ===")
g = Insight.ones(Int64[3], Insight.float32)
h = Insight.ones(Int64[3], Insight.float32)
i = Insight.add(g, h)
check("add defined", i.ptr != C_NULL)
check("add numel", Insight.numel(i) == 3)

j = Insight.sub(g, h)
check("sub defined", j.ptr != C_NULL)

k = Insight.mul(g, h)
check("mul defined", k.ptr != C_NULL)

# ===== Reduction =====
println("=== Reduction ===")
o = Insight.ones(Int64[3, 4], Insight.float32)
p = Insight.sum(o)
check("sum defined", p.ptr != C_NULL)
check("sum numel", Insight.numel(p) == 1)

q = Insight.mean(o)
check("mean defined", q.ptr != C_NULL)

# ===== Linear Algebra =====
println("=== Linalg ===")
r = Insight.ones(Int64[2, 3], Insight.float32)
s = Insight.ones(Int64[3, 4], Insight.float32)
t = Insight.matmul(r, s)
check("matmul defined", t.ptr != C_NULL)
check("matmul numel", Insight.numel(t) == 8)

u = Insight.eye(Int64(3), Int64(3), Insight.float32)
v = Insight.det(u)
check("det defined", v.ptr != C_NULL)

w = Insight.inv(u)
check("inv defined", w.ptr != C_NULL)

# ===== FFT =====
println("=== FFT ===")
x = Insight.ones(Int64[8], Insight.float32)
y = Insight.fft(x)
check("fft defined", y.ptr != C_NULL)

# ===== Display =====
println("=== Display ===")
println("Array repr: $(repr(a))")

# ===== Summary =====
println("")
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
