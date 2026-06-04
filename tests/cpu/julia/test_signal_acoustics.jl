# Signal acoustics CPU binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_signal_acoustics.jl

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

function approx(a, b; atol=1e-5)
    return Base.abs(Float64(a) - Float64(b)) < atol
end

# ============================================================================
# Acoustic Scale Conversions
# ============================================================================
println("=== Signal Acoustics ===")

# mel2hz
mel_arr = Insight.from_data([0.0, 1000.0, 2000.0])
hz_arr = Insight.mel2hz(mel_arr)
check("mel2hz", Insight.numel(hz_arr) == 3)

# hz2mel
hz_arr = Insight.from_data([0.0, 1000.0, 4000.0])
mel_arr = Insight.hz2mel(hz_arr)
check("hz2mel", Insight.numel(mel_arr) == 3)

# mel roundtrip
hz_orig = Insight.from_data([100.0, 500.0, 1000.0, 4000.0, 8000.0], Insight.float64)
mel = Insight.hz2mel(hz_orig)
hz_back = Insight.mel2hz(mel)
orig_data = Insight.to_data(hz_orig)
back_data = Insight.to_data(hz_back)
mel_ok = length(orig_data) == length(back_data)
if mel_ok
    for i in eachindex(orig_data)
        if !approx(orig_data[i], back_data[i], atol=1.0)
            local mel_ok = false; break
        end
    end
end
check("mel_roundtrip", mel_ok)

# mel_frequencies
freqs = Insight.mel_frequencies(128, f_low=0.0, f_high=11025.0)
check("mel_frequencies", Insight.numel(freqs) == 128)

# mel_frequencies custom
freqs = Insight.mel_frequencies(64, f_low=20.0, f_high=8000.0)
check("mel_frequencies_custom", Insight.numel(freqs) == 64)

# hz2bark
hz_arr = Insight.from_data([100.0, 500.0, 2000.0])
bark_arr = Insight.hz2bark(hz_arr)
check("hz2bark", Insight.numel(bark_arr) == 3)

# bark2hz
bark_arr = Insight.from_data([1.0, 5.0, 15.0])
hz_arr = Insight.bark2hz(bark_arr)
check("bark2hz", Insight.numel(hz_arr) == 3)

# bark roundtrip
hz_orig = Insight.from_data([100.0, 500.0, 2000.0, 6000.0], Insight.float64)
bark = Insight.hz2bark(hz_orig)
hz_back = Insight.bark2hz(bark)
orig_data = Insight.to_data(hz_orig)
back_data = Insight.to_data(hz_back)
bark_ok = length(orig_data) == length(back_data)
if bark_ok
    for i in eachindex(orig_data)
        if !approx(orig_data[i], back_data[i], atol=10.0)
            local bark_ok = false; break
        end
    end
end
check("bark_roundtrip", bark_ok)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
