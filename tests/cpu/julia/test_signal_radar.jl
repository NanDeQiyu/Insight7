# Signal radar CPU binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_signal_radar.jl

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

# ============================================================================
# Radar Signal Processing
# ============================================================================
println("=== Signal Radar ===")

# pulse_compression
n = 64
tpl_data = [sin(2 * pi * i / n) for i in 1:n]
sig_data = vcat(tpl_data, zeros(n))
tpl = Insight.from_data(tpl_data)
sig = Insight.from_data(sig_data)
result = Insight.pulse_compression(sig, tpl)
check("pulse_compression", Insight.numel(result) > 0)

# pulse_compression normalized
result = Insight.pulse_compression(sig, tpl, normalize=true)
check("pulse_compression_norm", Insight.numel(result) > 0)

# pulse_doppler
data_rows = [sin(2 * pi * r * c / 512) for r in 1:16, c in 1:32]
data_flat = vec(data_rows')
data_arr = Insight.from_data(collect(data_flat))
data_2d = Insight.reshape(data_arr, [32, 16])
result = Insight.pulse_doppler(data_2d)
check("pulse_doppler", Insight.numel(result) > 0)

# cfar_alpha
alpha = Insight.cfar_alpha(1e-3, 20)
check("cfar_alpha", alpha > 0.0)

# cfar_alpha different PFA
alpha1 = Insight.cfar_alpha(1e-2, 20)
alpha2 = Insight.cfar_alpha(1e-6, 20)
check("cfar_alpha_pfa", alpha1 < alpha2)

# mvdr
x_data = [sin(2 * pi * r * c / 128) for r in 1:4, c in 1:32]
x_flat = vec(x_data')
x_arr = Insight.from_data(collect(x_flat))
x_2d = Insight.reshape(x_arr, [32, 4])
sv = Insight.from_data([1.0, 1.0, 1.0, 1.0])
result = Insight.mvdr(x_2d, sv)
check("mvdr", Insight.numel(result) > 0)

# mvdr with different steering
sv2 = Insight.from_data([1.0, 0.5, 0.5, 1.0])
result2 = Insight.mvdr(x_2d, sv2)
check("mvdr_steering", Insight.numel(result2) > 0)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
