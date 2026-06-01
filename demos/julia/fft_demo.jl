# demos/julia/fft_demo.jl
# Demonstrates: FFT signal processing on CPU and GPU.

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

function run_fft_cpu()
    separator("CPU FFT")

    # Generate a signal: sum of two sinusoids
    n = 64
    t_vals = collect(0:n-1) ./ n
    signal = sin.(2π * 5 .* t_vals) .+ 0.5 .* sin.(2π * 12 .* t_vals)

    x = Insight.from_data(Float32.(signal), [n])
    first8 = [string(round(x[i], digits=3)) for i in 1:8]
    println("Input signal (first 8): " * join(first8, " ") * " ...")

    # RFFT -> IRFFT roundtrip
    X = Insight.rfft(x)
    println("RFFT output length: $(Insight.numel(X)) (complex)")

    x_recon = Insight.irfft(X, n)
    recon8 = [string(round(x_recon[i], digits=3)) for i in 1:8]
    println("Reconstructed signal (first 8): " * join(recon8, " ") * " ...")

    # Check reconstruction error
    max_err = maximum(abs.([x[i] - x_recon[i] for i in 1:n]))
    println("Max reconstruction error: $max_err")

    # F64 FFT
    separator("CPU FFT (F64)")
    signal_f64 = cos.(2π * 3 .* collect(0:n-1) ./ n)
    x64 = Insight.from_data(signal_f64, [n])
    X64 = Insight.rfft(x64)
    x64_recon = Insight.irfft(X64, n)
    max_err64 = maximum(abs.([signal_f64[i] - x64_recon[i] for i in 1:n]))
    println("F64 FFT roundtrip max error: $max_err64")

    # next_fast_len
    separator("FFT Length Optimization")
    println("next_fast_len(64)  = $(Insight.next_fast_len(64))")
    println("next_fast_len(100) = $(Insight.next_fast_len(100))")
    println("next_fast_len(127) = $(Insight.next_fast_len(127))")
end

function run_fft_gpu()
    separator("GPU FFT")

    n = 64
    t_vals = collect(0:n-1) ./ n
    signal = sin.(2π * 5 .* t_vals) .+ 0.5 .* sin.(2π * 12 .* t_vals)

    x_cpu = Insight.from_data(Float32.(signal), [n])
    x = Insight.to(x_cpu, 1)  # GPUPlace
    X = Insight.rfft(x)
    x_recon = Insight.to(Insight.irfft(X, n), 0)  # CPUPlace

    recon8 = [string(round(x_recon[i], digits=3)) for i in 1:8]
    println("GPU RFFT->IRFFT roundtrip (first 8): " * join(recon8, " ") * " ...")

    max_err = maximum(abs.([Float64(signal[i]) - x_recon[i] for i in 1:n]))
    println("GPU max reconstruction error: $max_err")

    # GPU F64 FFT
    separator("GPU FFT (F64)")
    signal_f64 = cos.(2π * 3 .* collect(0:n-1) ./ n)
    x64_cpu = Insight.from_data(signal_f64, [n])
    x64 = Insight.to(x64_cpu, 1)
    X64 = Insight.rfft(x64)
    x64_recon = Insight.to(Insight.irfft(X64, n), 0)
    max_err64 = maximum(abs.([signal_f64[i] - x64_recon[i] for i in 1:n]))
    println("GPU F64 FFT roundtrip max error: $max_err64")
end

try
    Insight.init(["cpu", "cuda"])
catch
    Insight.init(["cpu"])
end

println("Insight7 FFT Demo (Julia)")

run_fft_cpu()

if gpu_available()
    run_fft_gpu()
else
    println("\n[GPU not available, skipping GPU FFT demo]")
end

println("\nDone!")
