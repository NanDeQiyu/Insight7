# demos/julia/basic_ops.jl
# Demonstrates: array creation, arithmetic, multi-dtype, broadcasting,
# unary operations, and reductions.

# Add bindings to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "build", "bindings", "julia"))

using Insight

function separator(title)
    println("\n" * "="^40)
    println("  $title")
    println("="^40)
end

println("Insight7 Basic Operations Demo (Julia)")

# --- Array creation ---
separator("Array Creation")

a = Insight.ones([3, 4], Insight.float32)
println("ones(3,4) shape=($(Insight.shape(a))) dtype=$(Insight.dtype(a))")

b = Insight.arange(0.0, 12.0, 1.0, Insight.float64)
b = Insight.reshape(b, [3, 4])
println("arange(0,12).reshape(3,4):")
println(b)

c = Insight.eye(Int64(4), Int64(4), Insight.float64)
println("eye(4) F64:")
println(c)

# --- Multi-dtype arithmetic ---
separator("Multi-Dtype Arithmetic")

# F32
f32_a = Insight.cast(Insight.from_data([1.0, 2.0, 3.0]), Insight.float32)
f32_b = Insight.cast(Insight.from_data([4.0, 5.0, 6.0]), Insight.float32)
f32_c = f32_a + f32_b
println("F32: [1,2,3] + [4,5,6] = [$(f32_c[1]),$(f32_c[2]),$(f32_c[3])]")

# F64
f64_a = Insight.from_data([1.0, 2.0, 3.0], Insight.float64)
f64_b = Insight.from_data([0.5, 0.5, 0.5], Insight.float64)
f64_c = f64_a * f64_b
println("F64: [1,2,3] * [0.5,0.5,0.5] = [$(f64_c[1]),$(f64_c[2]),$(f64_c[3])]")

# I32
i32_a = Insight.cast(Insight.from_data([10.0, 20.0, 30.0]), Insight.int32)
i32_b = Insight.cast(Insight.from_data([3.0, 3.0, 3.0]), Insight.int32)
i32_c = i32_a / i32_b
println("I32: [10,20,30] / [3,3,3] = [$(i32_c[1]),$(i32_c[2]),$(i32_c[3])]")

# --- Broadcasting ---
separator("Broadcasting")

m = Insight.from_data(reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3), Insight.float64)
row = Insight.from_data(reshape([10.0, 20.0, 30.0], 1, 3), Insight.float64)
result = m + row
println("Matrix + Row broadcast:")
println(result)

# --- Unary operations ---
separator("Unary Operations")

x = Insight.from_data([-2.0, -1.0, 0.0, 1.0, 2.0], Insight.float64)
println("x:      $x")
println("abs(x): $(Insight.abs_fn(x))")
println("sin(x): $(Insight.sin_fn(x))")
println("exp(x): $(Insight.exp_fn(x))")

# --- Reductions ---
separator("Reductions")

data = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0], Insight.float64)
println("data:  $data")
println("sum:   $(Insight.sum(data))")
println("mean:  $(Insight.mean(data))")

println("\nDone!")
