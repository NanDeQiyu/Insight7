# Unary CPU tests — aligned with C++ test_unary.cpp
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_unary.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

passed = 0; failed = 0
function check(name, cond)
    global passed, failed
    if cond; passed += 1; else; failed += 1; println("FAIL: $name"); end
end

println("=== Unary ===")

u = Insight.from_data([-3.0, -1.0, 0.0, 2.0, 4.0])

# abs
check("abs", Insight.abs_fn(u).ptr != C_NULL)

# sqrt
v = Insight.from_data([1.0, 4.0, 9.0, 16.0])
check("sqrt", Insight.sqrt_fn(v).ptr != C_NULL)

# exp
w = Insight.from_data([0.0, 1.0, 2.0])
check("exp", Insight.exp_fn(w).ptr != C_NULL)

# log
check("log", Insight.log_fn(w).ptr != C_NULL)

# sin
x = Insight.from_data([0.0, 0.5, 1.0])
check("sin", Insight.sin_fn(x).ptr != C_NULL)

# cos
check("cos", Insight.cos_fn(x).ptr != C_NULL)

# tan
check("tan", Insight.tan_fn(x).ptr != C_NULL)

# floor
y = Insight.from_data([1.2, 2.5, 3.7, -1.3])
check("floor", Insight.floor_fn(y).ptr != C_NULL)

# ceil
check("ceil", Insight.ceil_fn(y).ptr != C_NULL)

# round
check("round", Insight.round_fn(y).ptr != C_NULL)

# negative
check("negative", Insight.negative(u).ptr != C_NULL)

# exp2
z = Insight.linspace(0.1, 2.0, Int64(10), Insight.float32)
check("exp2", Insight.exp2_fn(z).ptr != C_NULL)

# expm1
check("expm1", Insight.expm1_fn(z).ptr != C_NULL)

# log1p
check("log1p", Insight.log1p_fn(z).ptr != C_NULL)

# cbrt
check("cbrt", Insight.cbrt_fn(z).ptr != C_NULL)

# reciprocal
check("reciprocal", Insight.reciprocal_fn(z).ptr != C_NULL)

# asinh
check("asinh", Insight.asinh_fn(z).ptr != C_NULL)

# acosh
w2 = Insight.from_data([1.5, 2.0, 3.0])
check("acosh", Insight.acosh_fn(w2).ptr != C_NULL)

# deg2rad
d = Insight.from_data([0.0, 90.0, 180.0, 360.0])
check("deg2rad", Insight.deg2rad_fn(d).ptr != C_NULL)

# rad2deg
r = Insight.from_data([0.0, 1.5707963, 3.1415926, 6.2831853])
check("rad2deg", Insight.rad2deg_fn(r).ptr != C_NULL)

println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0; exit(1); end
