---
name: fix-cross-language-demo-gotchas
description: Common issues in Python/Lua/Julia demos — numel/item/tostring/init patterns
source: auto-skill
extracted_at: '2026-06-03T15:00:00.000Z'
---

# Cross-Language Demo Gotchas

## Python demos

### `numel` is a method, not a property
```python
# ❌ WRONG
freq_bins = spectrum.numel  # returns bound method, not int

# ✅ CORRECT
freq_bins = spectrum.numel()
```

### `item()` doesn't exist on Array
```python
# ❌ WRONG
val = ins.sum(arr).item()  # AttributeError

# ✅ CORRECT
val = ins.sum(arr).numpy().item()
```

### Array printing — use `str()` not `.numpy()`
```python
# ❌ WRONG — bypasses C++ tostring, shows numpy format
print(arr.numpy())
# [[ 0.  1.  2.  3.]
#  [ 4.  5.  6.  7.]]

# ✅ CORRECT — uses C++ insight_array_tostring(), shows Insight format
print(arr)
# Array(shape=[2, 4], dtype=float64, place=cpu:0,
#       [[0., 1., 2., 3.],
#        [4., 5., 6., 7.]])

# For energy/stats, use .numpy() to extract scalar value:
energy = float(ins.sum(arr).numpy())  # ✅ CORRECT

# ❌ WRONG — str() of Array includes metadata, float() fails
energy = float(str(ins.sum(arr)))  # ValueError: could not convert string
```

### Init — use auto-discover for future hardware support
```python
# ✅ CORRECT — auto-discover: CPU + first GPU if available
# On new hardware (Ascend, Cambricon, etc.), init() will find libinsight_*_backend.so
ins.init()

# Also OK but limits to CPU only:
ins.init(["cpu"])
```

**Why auto-discover**: After CI passes, the project will run on national hardware.
`init()` scans for `libinsight_*_backend.so` and loads whatever is available.
GPU sections in demos must have try-catch since GPU may not be present.

## Lua demos

### Array print shows full C++ format
```lua
-- The __tostring metamethod calls ins::to_string() via C API
-- Shows: Array(shape=[3, 4], dtype=float32, place=cpu:0,
--               [[0., 1., 2., 3.],
--                [4., 5., 6., 7.]])
print(tostring(arr))
```

### Scalar extraction
```lua
-- ❌ WRONG — ins.item doesn't exist as module function
local v = ins.item(arr, 0)

-- ❌ WRONG — arr[i] returns Array slice, not number
local v = arr[1]

-- ✅ CORRECT
local v = arr:get(0)  -- returns double at 0-based index
```

### FFT functions
```lua
-- ❌ WRONG — ins.fft is a function, not a namespace
local spectrum = ins.fft.rfft(signal)

-- ✅ CORRECT — rfft/irfft are top-level functions
local spectrum = ins.rfft(signal)
local filtered = ins.irfft(spectrum, frames)
```

### write_bin / read_bin argument order
```lua
-- The Lua wrapper matches Python convention: (filepath, data)
ins.signal.write_bin("/tmp/output.bin", signal)
local back = ins.signal.read_bin("/tmp/output.bin")
```

### Lua `from_table` creates float64 by default
```lua
-- ❌ WRONG — Lua numbers are float64, write_bin produces 8 bytes/element
local signal = ins.from_table(data)
ins.signal.write_bin(path, signal)  -- 88200 elements × 8 bytes = 705600 bytes

-- ✅ CORRECT — convert to float32 for consistent binary I/O (4 bytes/element)
local signal = ins.from_table(data):to(ins.float32)
ins.signal.write_bin(path, signal)  -- 88200 elements × 4 bytes = 352800 bytes
```

**Why:** Python creates float32 arrays by default. If Lua writes float64,
the roundtrip read count will be 2× what Python shows. Always match dtype.

### Comparison operators
```lua
-- ❌ WRONG — returns native Lua boolean
local c = (a == b)

-- ✅ CORRECT — returns Array
local c = ins.equal(a, b)
```

## Julia demos

### No `init` function
```julia
# ❌ WRONG
Insight.init(["cpu"])  # UndefVarError

# ✅ CORRECT — just use the module
using Insight
```

### No `getindex` on InsightArray
```julia
# ❌ WRONG
val = arr[1]  # MethodError

# ✅ CORRECT
val = Insight.item(arr, 0)
# or
data = Insight.to_data(arr)
val = data[1]
```

### `from_data` shape inference
```julia
# ❌ WRONG — [2,2] interpreted as dtype
A = Insight.from_data([1.0, 2.0, 3.0, 4.0], [2, 2])

# ✅ CORRECT
A = Insight.from_data(reshape([1.0, 2.0, 3.0, 4.0], 2, 2))
```

### `Base.abs` shadowed
```julia
# ❌ WRONG
max_err = maximum(abs.(x .- y))  # UndefVarError

# ✅ CORRECT
max_err = maximum(Base.abs.(x .- y))
```

### Other shadowed Base functions
`using Insight` also shadows `div`, `max`, `min`, `sin`, `cos`, `exp`, `log`, `ones`, `zeros`:
```julia
# ❌ WRONG
nyquist = div(sample_rate, 2)   # Insight.div (element-wise division)
val = max(0.0, min(1.0, t))     # Insight.max/min (element-wise)

# ✅ CORRECT
nyquist = Base.div(sample_rate, 2)
val = Base.max(0.0, Base.min(1.0, t))
```

### `read_bin` needs dtype keyword
```julia
# ❌ WRONG — reads as raw U8 bytes
data = Insight.read_bin("file.bin")  # numel = bytes, not elements

# ✅ CORRECT — specify dtype
data = Insight.read_bin("file.bin", dtype=Float32)
```

### `item()` requires index argument
```julia
# ❌ WRONG — no method matching item(::InsightArray)
val = Insight.item(scalar_arr)

# ✅ CORRECT — 0-based index
val = Insight.item(scalar_arr, 0)
```

### `fft`/`rfft` distinction
```julia
# Insight.fft does full complex FFT (input must be complex or auto-converted)
# Insight.rfft does real-to-complex FFT (returns n/2+1 complex values)
# Insight.irfft does inverse real FFT (complex input, real output)
X = Insight.rfft(signal)
x = Insight.irfft(X, original_length)
```

### No `load_backend` in Julia
```julia
# ❌ WRONG — UndefVarError
Insight.load_backend("cuda")

# GPU not available through Julia binding (no load_backend function)
# Julia demos should gracefully skip GPU sections
```

### `load_backend` exists and returns Bool
```julia
# load_backend DOES exist and returns Bool (true=success, false=failure)
# It does NOT throw on failure

# ❌ WRONG — always returns true
function gpu_available()
    try
        Insight.load_backend("cuda")
        return true
    catch
        return false
    end
end

# ✅ CORRECT — check return value
function gpu_available()
    try
        return Insight.load_backend("cuda")
    catch
        return false
    end
end
```

## GPU section pattern (ALL languages)

`gpu_available()` using `load_backend("cuda")` is unreliable — `load_backend` silently
fails when the .so doesn't exist (returns void/true), so `gpu_available()` returns `true`
even without a GPU. The actual failure happens later when `.to(GPUPlace(0))` is called.

**Fix**: Always wrap GPU section calls in try-catch, even when `gpu_available()` returns true.

```python
# Python
if gpu_available():
    try:
        run_gpu_linalg()
    except Exception as e:
        print(f"\n[GPU not available: {e}]")
else:
    print("\n[GPU not available, skipping GPU linalg demo]")
```

```lua
-- Lua
if gpu_available() then
  local ok, err = pcall(run_gpu_linalg)
  if not ok then
    print("\n[GPU not available: " .. tostring(err) .. "]")
  end
else
  print("\n[GPU not available, skipping GPU linalg demo]")
end
```

```julia
# Julia
if gpu_available()
    try
        run_gpu_linalg()
    catch e
        println("\n[GPU not available: $e]")
    end
else
    println("\n[GPU not available, skipping GPU linalg demo]")
end
```

**Why:** In CI with `-DINSIGHT_WITH_CUDA=OFF`, `load_backend("cuda")` doesn't throw
(it just fails to find the .so). The Lua/Python wrappers return `true` (success).
Then `GPUPlace(0)` → `to(GPUPlace(0))` throws because no GPU backend is registered.

**Symptom**: `RuntimeError: GPUPlace: GPU backend is not available` (Python)
or `luajit: C++ exception` (Lua) or `signal 6: Aborted` (Julia).

**How to apply**: Every demo with a GPU section must have try-catch around the call,
not just a `gpu_available()` check. This applies to linalg_demo, fft_demo, and any
future demo with GPU operations.

## C++ demos

### Guard plot calls with `#ifdef INSIGHT_USE_MATPLOT`
```cpp
// ❌ WRONG — fails when matplot is disabled
#include "insight/ops/plot.h"
void save_plots(...) { plot::figure(); ... }

// ✅ CORRECT
#ifdef INSIGHT_USE_MATPLOT
#include "insight/ops/plot.h"
void save_plots(...) { plot::figure(); ... }
#endif

// In main:
#ifdef INSIGHT_USE_MATPLOT
  save_plots(result, "prefix");
#endif
```

### GPU init with fallback
```cpp
// ❌ WRONG — crashes without CUDA
ins::init({"cpu", "cuda"});

// ✅ CORRECT
try { ins::init({"cpu", "cuda"}); }
catch (...) { ins::init({"cpu"}); }
```

## CI Demo Testing Pattern

```yaml
# Don't mask failures with || echo
# ❌ WRONG
- run: ./demo || echo "WARN: failed"

# ✅ CORRECT — let it fail
- run: ./demo
```

## Radar Demo Cross-Language Alignment

All languages must use Insight's FFT API (not numpy/FFTW directly) for consistent results.

### Lua complex pulse compression
```lua
-- ❌ WRONG — separate real/imag FFTs produce wrong results
local rr = ins.signal.fftconvolve(pulse_r_arr, mf_r_arr, "full")
local ii = ins.signal.fftconvolve(pulse_i_arr, mf_i_arr, "full")
-- ... 4 separate convolutions

-- ✅ CORRECT — use complex arrays with single fftconvolve
local pulse_c = ins.to_complex(pulse_r_arr, pulse_i_arr)
local mf_c = ins.to_complex(mf_r_arr, mf_i_arr)
local conv = ins.signal.fftconvolve(pulse_c, mf_c, "full")
local conv_r = ins.real(conv)
local conv_i = ins.imag(conv)
```

### Lua batched 2D FFT
```lua
-- ❌ WRONG — column-by-column FFT (extremely slow, 730s for 128 pulses)
for i = 1, N do
  local fft_r = ins.fft(col_r_arr)
  -- ...
end

-- ✅ CORRECT — build 2D complex array, batched FFT along axis 0
local pc_c = ins.to_complex(ins.from_table(pc_r_2d), ins.from_table(pc_i_2d))
local doppler_fft = ins.fftshift(ins.fft(pc_c, N_PULSES, 0), 0)
local energy = ins.sqrt(ins.add(ins.mul(ins.real(doppler_fft), ins.real(doppler_fft)),
                                 ins.mul(ins.imag(doppler_fft), ins.imag(doppler_fft))))
```

### Lua RNG overflow
```lua
-- ❌ WRONG — Lua numbers are double, can't store LCG large integers
local rng_state = 42
local function randn()
  rng_state = rng_state * 6364136223846793005 + 1442695040888963407  -- OVERFLOW!
  ...
end

-- ✅ CORRECT — use Lua's built-in RNG
math.randomseed(42)
local function randn()
  local u1 = math.random()
  local u2 = math.random()
  return math.sqrt(-2 * math.log(u1 + 1e-15)) * math.cos(2 * math.pi * u2)
end
```

### Python: use Insight FFT, not numpy
```python
# ❌ WRONG — uses numpy FFT, different from C++ Insight FFT
doppler_np = np.fft.fftshift(doppler_fft.numpy(), axes=0)
energy = np.abs(doppler_np)

# ✅ CORRECT — use Insight FFT functions
doppler_shifted = ins.fftshift(doppler_fft, 0)
energy_arr = ins.abs(doppler_shifted)
```

### Julia column-major `from_data` — PROPER FIX IMPLEMENTED

`from_data` now reverses dims automatically: Julia `(128, 1000)` → Insight `(1000, 128)`.
All axis parameters use `_julia_axis(arr, k) = ndim - k` for conversion.
This means batched 2D FFT works correctly without column-by-column workaround.

For CFAR and other 2D operations, use `permutedims` before `from_data`:
```julia
energy_ins = Insight.from_data(Base.permutedims(energy))
det = Insight.signal.ca_cfar(energy_ins, Int32[2,2], Int32[4,4], 1e-5)
det_jl = Base.permutedims(Base.reshape(Insight.to_data(det), N, N_PULSES))
```

For noise generation, use 1D arrays to avoid layout issues:
```julia
Insight.seed(42)
noise_flat = Insight.to_data(Insight.randn(Int64[N_PULSES * N], Insight.float64) * sigma)
# Access: noise_flat[(p-1)*N+1 : p*N] for pulse p
```

**Why NOT column-by-column FFT**: 730s for 128 pulses vs 0.03s batched. The layout
fix makes batched FFT correct, so column-by-column is no longer needed.

### Julia 1-based indexing for range bins
```julia
# ❌ WRONG — Julia findall returns 1-based indices
range_m = (r_idx - PC_OFFSET) * RANGE_PER_BIN  # off by 1 bin!

# ✅ CORRECT — convert to 0-based first
range_m = (r_idx - 1 - PC_OFFSET) * RANGE_PER_BIN
```

### Julia fftfreq parameter
```julia
# ❌ WRONG — passes sample rate instead of sample spacing
freqs = Insight.fftfreq(Int64(N_PULSES), 1.0/T_PRF)  # 10000 Hz, too large!

# ✅ CORRECT — pass sample spacing (T_PRF)
freqs = Insight.fftfreq(Int64(N_PULSES), T_PRF)  # 1e-4 seconds
```

## Plot Test Strategy (no gnuplot dependency)

```lua
-- ❌ WRONG — pending when gnuplot not installed (13 pending tests)
local function require_gnuplot()
  if not gnuplot_available then
    pending("gnuplot not installed, skipping plot tests")
  end
end

-- ✅ CORRECT — verify binding functions exist (14 tests, 0 pending)
it("plot function exists", function()
  assert.is_function(ins.plot.plot)
end)
```

## Pre-commit Before Every Commit

```bash
PATH="$HOME/.local/bin:$PATH" pre-commit run --all-files
# If files modified, re-stage and re-run
git add -u
PATH="$HOME/.local/bin:$PATH" pre-commit run --all-files
```

## OMP_NUM_THREADS=1 for Fast CPU Tests

```bash
OMP_NUM_THREADS=1 ctest -j24  # Much faster for small data
OMP_NUM_THREADS=1 ./tests/insight_tests_cpu  # 88ms vs seconds
```
