# Device information CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_device_info.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

# Try GPU
try
    Insight.load_backend("cuda")
    Insight.set_device(Insight.GPUPlace(0))
catch e
    println("SKIP: CUDA backend not available: $e")
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

println("=== Device Info CUDA ===")

# device_name gpu
name = Insight.device_name(0)
check("device_name", typeof(name) == String && length(name) > 0)

# cuda_version positive
ver = Insight.cuda_version()
check("cuda_version_positive", ver > 0)

# driver_version positive
dver = Insight.driver_version()
check("driver_version_positive", dver > 0)

# compute_capability positive
cc = Insight.compute_capability(0)
check("compute_capability_positive", cc > 0)

# gpu_count positive
gc = Insight.gpu_count()
check("gpu_count_positive", gc > 0)

# device_memory
mem = Insight.device_memory(0)
check("device_memory", mem.total > 0 && mem.free > 0 && mem.total >= mem.free)

# compute_capability range
check("compute_capability_range", 30 <= cc <= 100)

# cuda_version format
major = div(ver, 1000)
check("cuda_version_format", major >= 11)

# device_name stable
name2 = Insight.device_name(0)
check("device_name_stable", name == name2)

# compute_capability stable
cc2 = Insight.compute_capability(0)
check("compute_capability_stable", cc == cc2)

# device_memory total reasonable
check("device_memory_total", mem.total >= 1024 * 1024 * 1024)

# driver_version format
dmajor = div(dver, 1000)
check("driver_version_format", dmajor >= 11)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
