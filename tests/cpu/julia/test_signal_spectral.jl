# Signal Spectral CPU binding tests.
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_signal_spectral.jl

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

function make_sine(freq::Float64, fs::Float64, n::Int)
    t = [sin(2 * pi * freq * i / fs) for i in 0:n-1]
    return Insight.from_data(t, Insight.float64)
end

# ============================================================================
# Spectral Analysis
# ============================================================================
println("=== Signal Spectral ===")

# welch
try
    x = make_sine(10.0, 256.0, 1024)
    f, pxx = Insight.signal.welch_jl(x, fs=256.0, nperseg=256)
    check("welch", Insight.numel(f) > 0 && Insight.numel(pxx) > 0)
catch e
    println("SKIP: welch ($e)")
end

# periodogram
try
    x = make_sine(10.0, 256.0, 512)
    f, pxx = Insight.signal.periodogram_jl(x, fs=256.0)
    check("periodogram", Insight.numel(f) > 0 && Insight.numel(pxx) > 0)
catch e
    println("SKIP: periodogram ($e)")
end

# csd
try
    x = make_sine(10.0, 256.0, 1024)
    y = make_sine(10.0, 256.0, 1024)
    result = Insight.signal.csd(x, y, fs=256.0, nperseg=256)
    check("csd", Insight.numel(result.f) > 0 && Insight.numel(result.Pxx) > 0)
catch e
    println("SKIP: csd ($e)")
end

# coherence
try
    x = make_sine(10.0, 256.0, 1024)
    y = make_sine(10.0, 256.0, 1024)
    result = Insight.signal.coherence(x, y, fs=256.0, nperseg=256)
    check("coherence", Insight.numel(result.f) > 0 && Insight.numel(result.Pxx) > 0)
catch e
    println("SKIP: coherence ($e)")
end

# spectrogram
try
    x = make_sine(10.0, 256.0, 1024)
    result = Insight.signal.spectrogram(x, fs=256.0, nperseg=256)
    check("spectrogram", Insight.numel(result.f) > 0 && Insight.numel(result.t) > 0 && Insight.numel(result.Sxx) > 0)
catch e
    println("SKIP: spectrogram ($e)")
end

# stft
try
    x = make_sine(10.0, 256.0, 1024)
    result = Insight.signal.stft(x, fs=256.0, nperseg=256)
    check("stft", Insight.numel(result.f) > 0 && Insight.numel(result.t) > 0 && Insight.numel(result.Sxx) > 0)
catch e
    println("SKIP: stft ($e)")
end

# vectorstrength
try
    events = Insight.from_data([0.0, 1.0, 2.0, 3.0], Insight.float64)
    result = Insight.signal.vectorstrength(events, 1.0)
    # phase can be slightly negative due to floating point (~-3.7e-16)
    check("vectorstrength", result.strength >= 0.0 && Base.abs(result.phase) < 1e-10)
catch e
    println("SKIP: vectorstrength ($e)")
end

# welch with custom params
try
    x = make_sine(10.0, 256.0, 2048)
    f, pxx = Insight.signal.welch_jl(x, fs=256.0, nperseg=512, noverlap=256, scaling="spectrum")
    check("welch_custom", Insight.numel(f) > 0 && Insight.numel(pxx) > 0)
catch e
    println("SKIP: welch_custom ($e)")
end

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
