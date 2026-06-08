#!/usr/bin/env julia
#=
Timer Demo — Verifies the Timer API works correctly.
=#
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "bindings", "julia", "src"))

using Insight
using Insight: Timer, timer_start, timer_stop, timer_elapsed_ms, timer_destroy

# CPU Timer
let
    t = Timer(0, 0)
    timer_start(t)
    sleep(0.005)
    timer_stop(t)
    ms = timer_elapsed_ms(t)
    println(string("CPU timer: ", round(ms, digits=3), " ms"))
    if ms < 0.0 || ms > 100.0
        error("CPU timer out of range: $ms")
    end
    timer_destroy(t)
end

# GPU Timer (if available)
if is_device_available(DeviceKindGPU)
    load_backend("cuda")
    t = Timer(1, 0)
    timer_start(t)
    a = ones([256, 256], DTypeF32, GPUPlace(0))
    b = full([256, 256], 2.0, DTypeF32, GPUPlace(0))
    c = add(a, b)
    timer_stop(t)
    ms = timer_elapsed_ms(t)
    println(string("GPU timer: ", round(ms, digits=3), " ms"))
    timer_destroy(t)
else
    println("GPU timer: SKIPPED (GPU not available)")
end

println("OK")
