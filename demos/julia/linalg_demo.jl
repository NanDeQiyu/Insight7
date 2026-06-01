# demos/julia/linalg_demo.jl
# Demonstrates: linear algebra on CPU and GPU.

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

function run_cpu_linalg()
    separator("CPU Linear Algebra")

    # MatMul F64
    A = Insight.from_data([1.0, 2.0, 3.0, 4.0], [2, 2])
    B = Insight.from_data([5.0, 6.0, 7.0, 8.0], [2, 2])
    C = Insight.matmul(A, B)
    println("MatMul F64:")
    println(C)

    # MatMul F32
    A32 = Insight.cast(A, Insight.float32)
    B32 = Insight.cast(B, Insight.float32)
    C32 = Insight.matmul(A32, B32)
    println("MatMul F32:")
    println(C32)

    # Determinant
    try
        println("det([[1,2],[3,4]]) = $(Insight.det(A))")
    catch
        println("det: skipped (requires OpenBLAS)")
    end

    # Inverse
    try
        A_inv = Insight.inv(A)
        println("inv([[1,2],[3,4]]):")
        println(A_inv)
        I_mat = Insight.matmul(A, A_inv)
        println("A * A_inv (should be identity):")
        println(I_mat)
    catch
        println("inv: skipped (requires OpenBLAS)")
    end

    # SVD
    try
        D = Insight.from_data([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], [3, 3])
        U, S, VT = Insight.svd(D, false)
        println("SVD singular values: $S")
    catch
        println("SVD: skipped (requires OpenBLAS)")
    end

    # Solve linear system
    try
        A3 = Insight.from_data([3.0, 2.0, -1.0, 2.0, -2.0, 4.0, -1.0, 0.5, -1.0], [3, 3])
        b = Insight.from_data([1.0, -2.0, 0.0], [3])
        x = Insight.solve(A3, b)
        println("Ax=b solution: $x")
    catch
        println("solve: skipped (requires OpenBLAS)")
    end
end

function run_gpu_linalg()
    separator("GPU Linear Algebra")

    A = Insight.from_data([1.0, 2.0, 3.0, 4.0], [2, 2])
    B = Insight.from_data([5.0, 6.0, 7.0, 8.0], [2, 2])
    A_gpu = Insight.to(A, 1)  # GPUPlace(0)
    B_gpu = Insight.to(B, 1)
    C = Insight.matmul(A_gpu, B_gpu)
    println("GPU MatMul F64:")
    println(Insight.to(C, 0))  # CPUPlace

    # GPU MatMul F32
    A32_gpu = Insight.to(Insight.cast(A, Insight.float32), 1)
    B32_gpu = Insight.to(Insight.cast(B, Insight.float32), 1)
    C32 = Insight.matmul(A32_gpu, B32_gpu)
    println("GPU MatMul F32:")
    println(Insight.to(C32, 0))

    try
        println("GPU det = $(Insight.det(A_gpu))")
    catch
        println("GPU det: skipped (requires OpenBLAS)")
    end
    try
        A_inv = Insight.inv(A_gpu)
        println("GPU inv:")
        println(Insight.to(A_inv, 0))
    catch
        println("GPU inv: skipped (requires OpenBLAS)")
    end

    D = Insight.from_data([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], [3, 3])
    D_gpu = Insight.cast(D, Insight.float32)
    D_gpu = Insight.to(D_gpu, 1)
    try
        U, S, VT = Insight.svd(D_gpu, false)
        println("GPU SVD singular values (F32): $(Insight.to(S, 0))")
    catch
        println("GPU SVD: skipped (requires OpenBLAS)")
    end
end

try
    Insight.init(["cpu", "cuda"])
catch
    Insight.init(["cpu"])
end

println("Insight7 Linear Algebra Demo (Julia)")

run_cpu_linalg()

if gpu_available()
    run_gpu_linalg()
else
    println("\n[GPU not available, skipping GPU linalg demo]")
end

println("\nDone!")
