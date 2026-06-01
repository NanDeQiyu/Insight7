# demos/julia/sndfile_demo.jl
#
# Signal processing demo: generate a multi-frequency test signal,
# apply a low-pass filter via FFT, and save results.
# Mirrors the C++ sndfile_demo without libsndfile dependency.

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "bindings", "julia"))
using Insight

function separator(title)
    println("=" ^ 60)
    println("  $title")
    println("=" ^ 60)
end

Insight.init(["cpu"])

# Parameters
sample_rate = 44100
duration = 2.0
frames = Int(sample_rate * duration)

println("=== Insight Signal Processing Demo (Julia) ===")
println("    Sample rate: $sample_rate Hz")
println("    Duration:    $duration s ($frames frames)")
println()

# Step 1: Generate multi-frequency test signal
separator("[1] Generate test signal")
println("    Components: 440Hz (0.5), 880Hz (0.3), 3141Hz (0.1)")

signal_data = Float32[0.5 * sin(2π * 440 * i / sample_rate) +
                       0.3 * sin(2π * 880 * i / sample_rate) +
                       0.1 * sin(2π * 3141 * i / sample_rate)
                       for i in 0:frames-1]

signal = Insight.from_data(signal_data, [frames])
println("    Signal: [$(frames)] elements")
println()

# Step 2: Write signal to binary file
separator("[2] Save signal to binary file")
tmp = "/tmp/insight_demo_"

Insight.write_bin(tmp * "original.bin", signal)
println("    Written: $(tmp)original.bin")

read_back = Insight.read_bin(tmp * "original.bin")
println("    Roundtrip read: $(Insight.numel(read_back)) elements")
println()

# Step 3: FFT analysis
separator("[3] FFT analysis")

spectrum = Insight.fft(signal)
freq_bins = Insight.numel(spectrum)
println("    FFT bins: $freq_bins")
println("    Nyquist: $(div(sample_rate, 2)) Hz")
println("    Bin resolution: $(round(sample_rate / freq_bins, digits=2)) Hz")
println()

# Step 4: Low-pass filter via FFT (cutoff 1000Hz)
separator("[4] Low-pass filter (cutoff 1000Hz)")

cutoff_bin = Int(floor(1000 / (sample_rate / 2.0) * freq_bins))
taper_width = Int(floor(freq_bins / 100))
println("    Cutoff bin: $cutoff_bin (1000Hz)")
println("    Taper width: $taper_width bins")

# Zero out frequencies above cutoff
# Work with magnitude: keep bins below cutoff, taper at cutoff
filtered_spectrum = Insight.fft(signal)

# Create a frequency-domain mask by manipulating the spectrum
# For simplicity, use rfft approach: keep only low-frequency bins
r_spectrum = Insight.rfft(signal)
r_bins = Insight.numel(r_spectrum)

# Build mask for real FFT
mask_data = Float32[]
for i in 0:r_bins-1
    if i < cutoff_bin - div(taper_width, 2)
        push!(mask_data, 1.0f0)
    elseif i < cutoff_bin + div(taper_width, 2)
        t_val = 1.0f0 - (i - cutoff_bin + div(taper_width, 2)) / taper_width
        push!(mask_data, max(0.0f0, min(1.0f0, t_val)))
    else
        push!(mask_data, 0.0f0)
    end
end

mask = Insight.from_data(mask_data, [r_bins])

# Apply mask to magnitude of rfft result
r_real = Insight.real_part(r_spectrum)
r_imag = Insight.imag_part(r_spectrum)
filt_real = Insight.mul(r_real, mask)
filt_imag = Insight.mul(r_imag, mask)
filt_spectrum = Insight.to_complex(filt_real, filt_imag)

# Inverse FFT
filtered = Insight.irfft(filt_spectrum, frames)
println("    Filtered signal: [$(Insight.numel(filtered))] elements")

# Step 5: Energy comparison
println()
separator("[5] Signal statistics")

energy_orig = Insight.sum(Insight.mul(signal, signal))
energy_filt = Insight.sum(Insight.mul(filtered, filtered))

eo = Insight.item(energy_orig)
ef = Insight.item(energy_filt)

println("    Energy (original): $eo")
println("    Energy (filtered): $ef")
if eo > 0
    println("    Retained: $(round(ef / eo * 100, digits=1))%")
end

# Write filtered signal
Insight.write_bin(tmp * "filtered.bin", filtered)
println()
println("Output files:")
println("  $(tmp)original.bin  - Original multi-freq signal")
println("  $(tmp)filtered.bin  - Low-pass filtered (<1000Hz)")
println()
println("Done!")
