-- demos/lua/sndfile_demo.lua
--
-- Signal processing demo: generate a multi-frequency test signal,
-- apply a low-pass filter via FFT, and save results.
-- Mirrors the C++ sndfile_demo without libsndfile dependency.

local ins = require("insight")
local math = math
local pi = math.pi
local sin = math.sin
local sqrt = math.sqrt
local fmt = string.format

local function separator(title)
    print(string.rep("=", 60))
    print("  " .. title)
    print(string.rep("=", 60))
end

ins.init({"cpu"})

-- Parameters
local sample_rate = 44100
local duration = 2.0
local frames = math.floor(sample_rate * duration)

print("=== Insight Signal Processing Demo (Lua) ===")
print(fmt("    Sample rate: %d Hz", sample_rate))
print(fmt("    Duration:    %.1f s (%d frames)", duration, frames))
print()

-- Step 1: Generate multi-frequency test signal
separator("[1] Generate test signal")
print("    Components: 440Hz (0.5), 880Hz (0.3), 3141Hz (0.1)")

local signal_data = {}
for i = 0, frames - 1 do
    local t = i / sample_rate
    signal_data[i + 1] = 0.5 * sin(2 * pi * 440 * t)
                        + 0.3 * sin(2 * pi * 880 * t)
                        + 0.1 * sin(2 * pi * 3141 * t)
end

local signal = ins.from_table(signal_data)
print(fmt("    Signal: [%d] elements", signal.numel))
print()

-- Step 2: Write signal to binary file
separator("[2] Save signal to binary file")
local tmp = "/tmp/insight_demo_"

ins.signal.write_bin(tmp .. "original.bin", signal)
print("    Written: " .. tmp .. "original.bin")

local read_back = ins.signal.read_bin(tmp .. "original.bin")
print(fmt("    Roundtrip read: %d elements", read_back.numel))
print()

-- Step 3: FFT analysis
separator("[3] FFT analysis")

local spectrum = ins.fft.rfft(signal)
local freq_bins = spectrum.numel
print(fmt("    FFT bins: %d", freq_bins))
print(fmt("    Nyquist: %d Hz", sample_rate / 2))
print(fmt("    Bin resolution: %.2f Hz", sample_rate / (2 * freq_bins)))
print()

-- Step 4: Low-pass filter via FFT (cutoff 1000Hz)
separator("[4] Low-pass filter (cutoff 1000Hz)")

local cutoff_bin = math.floor(1000 / (sample_rate / 2.0) * freq_bins)
local taper_width = math.floor(freq_bins / 100)
print(fmt("    Cutoff bin: %d (1000Hz)", cutoff_bin))
print(fmt("    Taper width: %d bins", taper_width))

-- Zero out frequencies above cutoff in the spectrum
-- Work with real and imaginary parts separately
local spec_real = ins.real(spectrum)
local spec_imag = ins.imag(spectrum)

-- Build a real-valued mask [0, 1] for each frequency bin
local mask_data = {}
for i = 0, freq_bins - 1 do
    if i < cutoff_bin - math.floor(taper_width / 2) then
        mask_data[i + 1] = 1.0
    elseif i < cutoff_bin + math.floor(taper_width / 2) then
        local t_val = 1.0 - (i - cutoff_bin + taper_width / 2.0) / taper_width
        mask_data[i + 1] = math.max(0.0, math.min(1.0, t_val))
    else
        mask_data[i + 1] = 0.0
    end
end

local mask = ins.from_table(mask_data)

-- Apply mask to real and imaginary parts
local filtered_real = ins.mul(spec_real, mask)
local filtered_imag = ins.mul(spec_imag, mask)

-- Reconstruct complex spectrum
local filtered_spectrum = ins.to_complex(filtered_real, filtered_imag)

-- Inverse FFT to get filtered signal
local filtered = ins.fft.irfft(filtered_spectrum, frames)
print(fmt("    Filtered signal: [%d] elements", filtered.numel))

-- Step 5: Energy comparison
print()
separator("[5] Signal statistics")

local energy_orig = ins.sum(ins.mul(signal, signal))
local energy_filt = ins.sum(ins.mul(filtered, filtered))

-- Extract scalar values
local eo = energy_orig:item()
local ef = energy_filt:item()

print(fmt("    Energy (original): %.4f", eo))
print(fmt("    Energy (filtered): %.4f", ef))
if eo > 0 then
    print(fmt("    Retained: %.1f%%", (ef / eo) * 100))
end

-- Write filtered signal
ins.signal.write_bin(tmp .. "filtered.bin", filtered)
print()
print("Output files:")
print("  " .. tmp .. "original.bin  - Original multi-freq signal")
print("  " .. tmp .. "filtered.bin  - Low-pass filtered (<1000Hz)")
print()
print("Done!")
