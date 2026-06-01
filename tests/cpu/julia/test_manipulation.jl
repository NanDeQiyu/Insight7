# Manipulation CPU tests — aligned with C++ test_manipulation.cpp
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_manipulation.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

passed = 0; failed = 0
function check(name, cond)
    global passed, failed
    if cond; passed += 1; else; failed += 1; println("FAIL: $name"); end
end

println("=== Manipulation ===")

# reshape
a = Insight.arange(0.0, 6.0, 1.0, Insight.float64)
b = Insight.reshape(a, Int64[2, 3])
check("reshape numel", Insight.numel(b) == 6)
check("reshape ndim", Insight.ndim(b) == 2)

# transpose
a = Insight.from_data(Float64[1 2 3; 4 5 6])
b = Insight.transpose(a)
check("transpose numel", Insight.numel(b) == 6)

# permute
a = Insight.ones(Int64[2, 3, 4], Insight.float64)
b = Insight.permute(a, Int32[2, 0, 1])
check("permute", Insight.numel(b) == 24)

# swapaxes
a = Insight.ones(Int64[2, 3], Insight.float64)
b = Insight.swapaxes(a, 0, 1)
check("swapaxes", Insight.numel(b) == 6)

# moveaxis
a = Insight.ones(Int64[2, 3, 4], Insight.float64)
b = Insight.moveaxis(a, 0, 2)
check("moveaxis", Insight.numel(b) == 24)

# fliplr
m2 = Insight.from_data(Float32[1 2 3; 4 5 6])
check("fliplr", Insight.fliplr(m2).ptr != C_NULL)

# flipud
check("flipud", Insight.flipud(m2).ptr != C_NULL)

# rot90
a2 = Insight.from_data(Float64[1 2; 3 4])
b = Insight.rot90(a2)
check("rot90", Insight.numel(b) == 4)

# diag_fn
d = Insight.from_data([1.0, 2.0, 3.0])
b = Insight.diag_fn(d)
check("diag_fn numel", Insight.numel(b) == 9)
check("diag_fn ndim", Insight.ndim(b) == 2)

# diagonal
a3 = Insight.from_data(Float64[1 2 3; 4 5 6; 7 8 9])
b = Insight.diagonal(a3)
check("diagonal", Insight.numel(b) == 3)

# tril
check("tril", Insight.tril(m2).ptr != C_NULL)

# triu
check("triu", Insight.triu(m2).ptr != C_NULL)

# diff_fn
a4 = Insight.from_data([1.0, 3.0, 6.0, 10.0])
b = Insight.diff_fn(a4)
check("diff_fn", Insight.numel(b) == 3)

# squeeze (via reshape - Julia doesn't have direct squeeze)
a = Insight.zeros(Int64[1, 3, 1], Insight.float64)
# squeeze is not directly available in Julia bindings
# Use reshape as alternative
b = Insight.reshape(a, Int64[3])
check("squeeze_via_reshape", Insight.numel(b) == 3)

println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0; exit(1); end
