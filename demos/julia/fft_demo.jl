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

    x = Insight.from_data(Float32.(signal))
    first8 = [string(round(Insight.item(x, i-1), digits=3)) for i in 1:8]
    println("Input signal (first 8): " * join(first8, " ") * " ...")

    # RFFT -> IRFFT roundtrip
    X = Insight.fft(x)
    println("FFT output length: $(Insight.numel(X)) ()")

    x_recon = Insight.ifft(X)
    recon8 = [string(round(Insight.item(x_recon, i-1), digits=3)) for i in 1:8]
    println("Reconstructed signal (first 8): " * join(recon8, " ") * " ...")

    # Check reconstruction error
    x_data = Insight.to_data(x)
    x_recon_data = Insight.to_data(x_recon)
    max_err = maximum(Base.abs.(x_data .- x_recon_data))
    println("Max reconstruction error: $max_err")

    # F64 FFT
    separator("CPU FFT (F64)")
    signal_f64 = cos.(2π * 3 .* collect(0:n-1) ./ n)
    x64 = Insight.from_data(signal_f64)
    X64 = Insight.fft(x64)
    x64_recon = Insight.ifft(X64)
    x64_data = Insight.to_data(x64)
    x64_recon_data = Insight.to_data(x64_recon)
    max_err64 = maximum(Base.abs.(x64_data .- x64_recon_data))
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

    x_cpu = Insight.from_data(Float32.(signal))
    x = Insight.to(x_cpu, 1)  # GPUPlace
    X = Insight.fft(x)
    x_recon = Insight.to(Insight.ifft(X), 0)  # CPUPlace

    recon8 = [string(round(Insight.item(x_recon, i-1), digits=3)) for i in 1:8]
    println("GPU FFT->IFFT roundtrip (first 8): " * join(recon8, " ") * " ...")

    x_recon_data = Insight.to_data(x_recon)
    signal_data = Float64.(signal)
    max_err = maximum(Base.abs.(signal_data .- Float64.(x_recon_data)))
    println("GPU max reconstruction error: $max_err")

    # GPU F64 FFT
    separator("GPU FFT (F64)")
    signal_f64 = cos.(2π * 3 .* collect(0:n-1) ./ n)
    x64_cpu = Insight.from_data(signal_f64)
    x64 = Insight.to(x64_cpu, 1)
    X64 = Insight.fft(x64)
    x64_recon = Insight.to(Insight.ifft(X64), 0)
    x64_recon_data = Insight.to_data(x64_recon)
    max_err64 = maximum(Base.abs.(signal_f64 .- x64_recon_data))
    println("GPU F64 FFT roundtrip max error: $max_err64")
end

try
catch
end

println("Insight7 FFT Demo (Julia)")

run_fft_cpu()

if gpu_available()
    run_fft_gpu()
else
    println("\n[GPU not available, skipping GPU FFT demo]")
end

println("\nDone!")
