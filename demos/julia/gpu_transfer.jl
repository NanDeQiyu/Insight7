# demos/julia/gpu_transfer.jl
# Demonstrates: GPU device management, CPU-GPU data transfer.

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "build", "bindings", "julia"))

using Insight

function separator(title)
    println("\n" * "="^40)
    println("  $title")
    println("="^40)
end

function gpu_available()
    try
        Insight.load_backend("cuda")
        return true
    catch
        return false
    end
end

try
    Insight.init(["cpu", "cuda"])
catch
    Insight.init(["cpu"])
end

println("Insight7 GPU Transfer Demo (Julia)")

if !gpu_available()
    println("GPU not available. This demo requires a CUDA device.")
    exit(0)
end

# --- CPU to GPU transfer ---
separator("CPU -> GPU Transfer")

cpu_arr = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
println("CPU array:")
println(cpu_arr)

gpu_arr = Insight.to(cpu_arr, 1)  # GPUPlace
println("GPU array shape=$(Insight.shape(gpu_arr)) place=$(Insight.device_type(gpu_arr) == 1 ? "gpu" : "cpu")")

# --- GPU to CPU transfer ---
separator("GPU -> CPU Transfer")

back = Insight.to(gpu_arr, 0)  # CPUPlace
println("Round-tripped back to CPU:")
println(back)

# --- GPU arithmetic ---
separator("GPU Arithmetic (F64)")

a = Insight.to(Insight.from_data([1.0, 2.0, 3.0], [3]), 1)
b = Insight.to(Insight.from_data([4.0, 5.0, 6.0], [3]), 1)

sum_ab = a + b
mul_ab = a * b
println("GPU [1,2,3] + [4,5,6] = $(Insight.to(sum_ab, 0))")
println("GPU [1,2,3] * [4,5,6] = $(Insight.to(mul_ab, 0))")

# --- GPU arithmetic F32 ---
separator("GPU Arithmetic (F32)")

a32 = Insight.to(Insight.cast(Insight.from_data([1.0, 2.0, 3.0], [3]), Insight.float32), 1)
b32 = Insight.to(Insight.cast(Insight.from_data([4.0, 5.0, 6.0], [3]), Insight.float32), 1)
c32 = a32 * b32 + a32
println("GPU F32: [1,2,3]*[4,5,6]+[1,2,3] = $(Insight.to(c32, 0))")

# --- GPU reductions ---
separator("GPU Reductions")

data = Insight.to(Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0], [5]), 1)
println("GPU sum:  $(Insight.sum(data))")
println("GPU mean: $(Insight.mean(data))")
println("GPU max:  $(Insight.max(data))")

# --- Zero-copy views ---
separator("Zero-Copy Views (GPU)")

base = Insight.reshape(Insight.arange(12.0, Insight.float32), [3, 4])
gpu_base = Insight.to(base, 1)

# Reshape
reshaped = Insight.reshape(gpu_base, [4, 3])
println("Reshape 3x4 -> 4x3: shape=$(Insight.shape(reshaped))")

# Transpose
transposed = Insight.transpose(gpu_base)
println("Transpose 3x4 -> 4x3: shape=$(Insight.shape(transposed))")

println("\nDone!")
