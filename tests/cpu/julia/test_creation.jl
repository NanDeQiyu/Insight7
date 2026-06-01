# Creation CPU tests — aligned with C++ test_creation.cpp
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_creation.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

passed = 0; failed = 0
function check(name, cond)
    global passed, failed
    if cond; passed += 1; else; failed += 1; println("FAIL: $name"); end
end

println("=== Creation ===")

# zeros_1d
a = Insight.zeros(Int64[5], Insight.float32)
check("zeros_1d", Insight.numel(a) == 5)

# zeros_2d
a = Insight.zeros(Int64[3, 4], Insight.float64)
check("zeros_2d numel", Insight.numel(a) == 12)
check("zeros_2d ndim", Insight.ndim(a) == 2)

# ones_1d
a = Insight.ones(Int64[4], Insight.float32)
check("ones_1d", Insight.numel(a) == 4)

# ones_2d
a = Insight.ones(Int64[2, 5], Insight.float64)
check("ones_2d", Insight.numel(a) == 10)

# full
a = Insight.full(Int64[3, 3], 7.0, Insight.float32)
check("full", Insight.numel(a) == 9)

# full_negative
a = Insight.full(Int64[2, 2], -3.5, Insight.float64)
check("full_negative", Insight.numel(a) == 4)

# eye
a = Insight.eye(Int64(4), Int64(4), Insight.float32)
check("eye", Insight.numel(a) == 16)

# arange
a = Insight.arange(0.0, 10.0, 1.0, Insight.float64)
check("arange", Insight.numel(a) == 10)

# linspace
a = Insight.linspace(0.0, 1.0, Int64(11), Insight.float64)
check("linspace", Insight.numel(a) == 11)

# from_data
a = Insight.from_data([1.5, 2.5, 3.5])
check("from_data", Insight.numel(a) == 3)

# from_data_2d
a = Insight.from_data(Float32[1 2 3; 4 5 6])
check("from_data_2d numel", Insight.numel(a) == 6)
check("from_data_2d ndim", Insight.ndim(a) == 2)

# dtype_float64
a = Insight.zeros(Int64[3], Insight.float64)
check("dtype_float64", Insight.dtype(a) == Insight.float64)

# dtype_int32
a = Insight.ones(Int64[3], Insight.int32)
check("dtype_int32", Insight.dtype(a) == Insight.int32)

println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0; exit(1); end
