---
name: optimize-cross-language-perf
description: Profile and optimize binding language (Lua/Julia/Python) demo performance to match or exceed C++ baseline
source: auto-skill
extracted_at: '2026-06-07T00:30:00.000Z'
---

# Cross-Language Performance Optimization

## Problem

Binding languages (Lua/LuaJIT, Julia, Python) are slower than C++ due to:
- sol2/pybind11 per-call overhead (~5-50μs per API call)
- GC/finalizer pressure (Julia: InsightArray with finalizer per ccall)
- String parsing overhead (Lua string slicing via sol2)
- Unnecessary intermediate array creation

## Diagnosis: Per-Section Timing Breakdown

Add timing around each pipeline section in ALL languages (C++/Python/Lua/Julia):

```
echo_ms    — simulate_echoes
pc_ms      — pulse_compression
doppler_ms — doppler processing (mean + window + FFT + fftshift)
cfar_ms    — CA-CFAR
ext_ms     — target extraction (nonzero + transfer + clustering)
```

Compare breakdowns across languages to find the slowest section in each language.

## Optimization Hierarchy (Most Impactful First)

### 1. Cache constants in init_cache (ALL languages)

Pre-compute anything that doesn't change between frames:
- `_TEMPLATE` — pulse compression template slice
- `_HAMMING` — window array reshaped to [N_PULSES, 1]
- `_SLOW_TIMES` — `arange(N_PULSES) * T_PRF`

**Must apply to ALL languages** for algorithm alignment.

### 2. Use nonzero + bulk extraction (Lua/Julia)

Instead of scanning all 32000 elements:
```lua
-- ❌ Slow: 32000 sol2 get() calls
for idx = 0, det.numel - 1 do
    if det:get(idx) ~= 0 then ... end
end

-- ✅ Fast: nonzero on GPU + bulk table()
local idx = ins.nonzero(det)
local doppler_list = idx[1]:table()  -- ~90 elements
local range_list = idx[2]:table()
local energy_cpu = energy:to(cpu)
for k = 1, #doppler_list do
    local v = energy_cpu:get(doppler_list[k] * N + range_list[k])
end
```

### 3. Use integer slice() instead of string slicing (Lua)

```lua
-- ❌ Slow: string concatenation + parsing per call
local src = _S_TX["1:" .. tostring(N - ds + 1)]

-- ✅ Fast: integer params, no parsing
local src = ins.slice(_S_TX, 1, 1, N - ds + 1)
```

Requires adding `m["slice"]` to Lua binding:
```cpp
m["slice"] = [](const Array &a, int dim, int64_t start, int64_t stop) {
    return a.slice(dim - 1, start - 1, stop - 1);  // 1-based → 0-based
};
```

### 4. Wrap bare function pointers with sol::optional (Lua)

Functions with C++ default parameters lose defaults when bound as bare pointers:
```cpp
// ❌ Loses defaults — sol2 requires all args
sig["pulse_doppler"] = &signal::pulse_doppler;

// ✅ Lambda preserves defaults
sig["pulse_doppler"] = [](const Array &x, sol::optional<std::string> window,
                           sol::optional<int64_t> nfft) {
    return signal::pulse_doppler(x, window.value_or(""), nfft.value_or(0));
};
```

### 5. Disable GC in hot loop (Julia)

```julia
GC.enable(false)
for frame in 0:n_frames-1
    # ... run_frame ...
end
GC.enable(true)
GC.gc()
```

This prevents finalizer overhead during the loop. Memory accumulates but is freed after.

### 6. Remove unnecessary contiguous() calls (C++ core)

If `fftshift`/`ifftshift` add `contiguous()` copies that the optimized concat kernel doesn't need:
```cpp
// ❌ Unnecessary — our concat handles non-contiguous views
Array last_contig = contiguous(last);
Array first_contig = contiguous(first);
Array result = concat({last_contig, first_contig}, ax);

// ✅ Direct — concat handles views via cudaMemcpy2D
Array result = concat({last, first}, ax);
```

## Post-Merge Recovery Checklist

After merging external PRs, check for:
1. `array_type["table"]` removed from Lua binding
2. `array_type["slice"]` removed from Lua binding  
3. Bare function pointers replacing lambda wrappers (ambgfun, pulse_doppler)
4. Unnecessary `contiguous()` calls added to core ops
5. `.shape()` → `.shape` API changes in Python demos
6. `build.luarocks/` accidentally committed (add to `.gitignore`)

## Lua Test Gotcha: Integer Index Returns 0-d Array

`y[1]` returns a 0-d Array (not a number). Tests must use `:item()`:
```lua
-- ❌ Wrong: compares Array vs number
assert.are.equal(5.0, y[1])

-- ✅ Correct: extract scalar first
assert.are.equal(5.0, y[1]:item())
```

## ambgfun/pulse_doppler Must Use Lambda Wrappers

`sol::optional<Array>` default parameters cause segfault when bound as bare function pointers:
```cpp
// ❌ CRASH: sol2 passes empty Array() which segfaults inside ambgfun
sig["ambgfun"] = &signal::ambgfun;

// ✅ Safe: explicit sol::optional with default
sig["ambgfun"] = [](const Array &x, double fs, double prf,
                     sol::optional<Array> y) {
    return signal::ambgfun(x, fs, prf, y.value_or(Array()));
};
```

## Performance Reference (A800 GPU, radar detection demo)

| Language | 50-frame avg | Frame 9 steady | Key bottleneck |
|----------|-------------|----------------|----------------|
| C++ | 4.45ms | 4.16ms | — |
| Python | 5.15ms | 4.51ms | pybind11 overhead |
| Lua/LuaJIT | 5.69ms | 4.59ms | sol2 overhead + extract |
| Julia | 8.45ms | 5.22ms | ccall finalizer GC |

## Key Insight (2026): Binding Layer Is the Only Differentiator

For GPU-heavy workloads where computation happens in C++/CUDA kernels:
- **Python/pybind11 ≈ LuaJIT/sol2** — both are thin wrappers, JIT doesn't help across C API boundary
- **Julia ccall** is structurally slower due to GC finalizer per `new Array()`
- **LuaJIT's JIT advantage** only applies to pure Lua computation (no C calls)
- **pybind11 wins** because: reference counting (instant destruct), 15+ years of optimization, no GC pause

Pure compute benchmark (fib(35), 1e8 sum, sin+cos 1e7): LuaJIT 10-60x faster than Python.
GPU framework benchmark (Insight7 radar): Python ≈ LuaJIT — binding layer is negligible.

**Conclusion**: For Insight7-style frameworks, optimize C++/CUDA kernels first. Language choice is an ecosystem decision, not a performance decision.
