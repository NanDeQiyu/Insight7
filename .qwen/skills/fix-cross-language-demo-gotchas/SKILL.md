---
name: fix-cross-language-demo-gotchas
description: Common issues in Python/Lua/Julia demos — numel/item/tostring/init patterns
source: auto-skill
extracted_at: '2026-06-02T06:00:00.000Z'
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

# For energy/stats, .numpy() is still correct for computation:
energy = float(str(ins.sum(arr)))  # or float(ins.sum(arr).numpy())
```

### Init with CPU only
```python
# ❌ WRONG — tries to load CUDA, prints warning
ins.init(["cpu", "cuda"])

# ✅ CORRECT — CPU only, no warning
ins.init(["cpu"])
```

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
