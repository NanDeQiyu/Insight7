-- demos/lua/fft_demo.lua
-- Demonstrates: FFT signal processing on CPU and GPU.

local ins = require("insight")

local function separator(title)
  print(string.rep("=", 40))
  print("  " .. title)
  print(string.rep("=", 40))
end

local function gpu_available()
  local ok, result = pcall(function()
    return ins.load_backend("cuda")
  end)
  return ok and result
end

local function run_fft_cpu()
  separator("CPU FFT")

  -- Generate a signal: sum of two sinusoids
  local n = 64
  local sig = {}
  for i = 0, n - 1 do
    local t = i / n
    sig[i + 1] = math.sin(2 * math.pi * 5 * t) + 0.5 * math.sin(2 * math.pi * 12 * t)
  end

  local x = ins.from_table(sig):to(ins.float32)
  local first8 = {}
  for i = 1, 8 do
    first8[i] = string.format("%.3f", x:get(i - 1))
  end
  print("Input signal (first 8): " .. table.concat(first8, " ") .. " ...")

  -- RFFT -> IRFFT roundtrip
  local X = ins.rfft(x)
  print(string.format("RFFT output length: %d (complex)", X.numel))

  local x_recon = ins.irfft(X, n)
  local recon8 = {}
  for i = 1, 8 do
    recon8[i] = string.format("%.3f", x_recon:get(i - 1))
  end
  print("Reconstructed signal (first 8): " .. table.concat(recon8, " ") .. " ...")

  -- Check reconstruction error
  local max_err = 0
  for i = 0, n - 1 do
    local err = math.abs(x:get(i) - x_recon:get(i))
    if err > max_err then
      max_err = err
    end
  end
  print(string.format("Max reconstruction error: %e", max_err))

  -- F64 FFT
  separator("CPU FFT (F64)")
  local sig64 = {}
  for i = 0, n - 1 do
    sig64[i + 1] = math.cos(2 * math.pi * 3 * i / n)
  end
  local x64 = ins.from_table(sig64)
  local X64 = ins.rfft(x64)
  local x64_recon = ins.irfft(X64, n)
  local max_err64 = 0
  for i = 0, n - 1 do
    local err = math.abs(sig64[i + 1] - x64_recon:get(i))
    if err > max_err64 then
      max_err64 = err
    end
  end
  print(string.format("F64 FFT roundtrip max error: %e", max_err64))

  -- next_fast_len
  separator("FFT Length Optimization")
  print(string.format("next_fast_len(64)  = %d", ins.next_fast_len(64)))
  print(string.format("next_fast_len(100) = %d", ins.next_fast_len(100)))
  print(string.format("next_fast_len(127) = %d", ins.next_fast_len(127)))
end

local function run_fft_gpu()
  separator("GPU FFT")

  local n = 64
  local sig = {}
  for i = 0, n - 1 do
    local t = i / n
    sig[i + 1] = math.sin(2 * math.pi * 5 * t) + 0.5 * math.sin(2 * math.pi * 12 * t)
  end

  local x_cpu = ins.from_table(sig):to(ins.float32)
  local x = x_cpu:to(ins.GPUPlace(0))
  local X = ins.rfft(x)
  local x_recon = ins.irfft(X, n):to(ins.CPUPlace())

  local recon8 = {}
  for i = 1, 8 do
    recon8[i] = string.format("%.3f", x_recon:get(i - 1))
  end
  print("GPU RFFT->IRFFT roundtrip (first 8): " .. table.concat(recon8, " ") .. " ...")

  local max_err = 0
  for i = 0, n - 1 do
    local err = math.abs(sig[i + 1] - x_recon:get(i))
    if err > max_err then
      max_err = err
    end
  end
  print(string.format("GPU max reconstruction error: %e", max_err))

  -- GPU F64 FFT
  separator("GPU FFT (F64)")
  local sig64 = {}
  for i = 0, n - 1 do
    sig64[i + 1] = math.cos(2 * math.pi * 3 * i / n)
  end
  local x64_cpu = ins.from_table(sig64)
  local x64 = x64_cpu:to(ins.GPUPlace(0))
  local X64 = ins.rfft(x64)
  local x64_recon = ins.irfft(X64, n):to(ins.CPUPlace())
  local max_err64 = 0
  for i = 0, n - 1 do
    local err = math.abs(sig64[i + 1] - x64_recon:get(i))
    if err > max_err64 then
      max_err64 = err
    end
  end
  print(string.format("GPU F64 FFT roundtrip max error: %e", max_err64))
end

ins.init({ "cpu" })

print("Insight7 FFT Demo (Lua)")

run_fft_cpu()

if gpu_available() then
  run_fft_gpu()
else
  print("\n[GPU not available, skipping GPU FFT demo]")
end

print("\nDone!")
