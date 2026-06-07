---
name: cross-language-demo-porting
description: Patterns for porting Python radar detection demo to C++, Lua, Julia — CLI args, framework APIs, 1-based indexing, missing bindings
source: auto-skill
extracted_at: '2026-06-06T22:24:57.582Z'
---

# Cross-Language Demo Porting Patterns

## Core Principle
Use the same algorithm across all languages: echo simulation → pulse_compression → pulse_doppler → ca_cfar → target extraction. Only the API syntax differs.

## Framework API Usage (all languages)
Use these core functions for Task 1:

| Step | API | Python | C++ | Lua | Julia |
|------|-----|--------|-----|-----|-------|
| Pulse compression | `pulse_compression(x, template)` | `ins.signal.pulse_compression()` | `signal::pulse_compression()` | `ins.signal.pulse_compression()` | `Insight.signal.pulse_compression()` |
| Doppler | `pulse_doppler(x, window, nfft)` | same | same | same | same |
| CFAR | `ca_cfar(energy, guard, ref, pfa)` | same | same | same (returns pair) | same (returns tuple) |
| Ambiguity fn | `ambgfun(x, fs, prf)` | pass `fs=, prf=` kwargs | `signal::ambgfun(x, fs, prf)` | `ins.signal.ambgfun(x, fs, prf)` | `Insight.signal.ambgfun(x, fs, prf)` |

## CLI Args
Each language uses its native CLI parsing:
- **C++**: manual `strcmp` loop over `argv` (no getopt dependency)
- **Lua**: iterate `arg` table (global, 1-indexed)
- **Julia**: iterate `ARGS` (global, 1-indexed)

Args: `--device cpu|gpu`, `--seed N`, `--iterations N`

## Per-Language Gotchas

### C++
- `seed(seed_val)` — variable name `seed` shadows the function; use `seed_val`
- `candidates.push_back({i, r})` — brace-init list not convertible to struct; construct explicitly: `TargetInfo ti; ti.d=i; ti.r=r; result.targets.push_back(ti);`
- `fft::fftshift`, `fft::fftfreq` — need `fft::` namespace prefix
- `Slice(start, stop)` — `Slice()` with no args = full slice `::`
- `array.slice({Slice(), Slice(start, stop)})` — use `slice()` method, not `operator[]`
- `copy_from_()` for assigning views — `view.copy_from_(src)`
- `mul(arr, scalar)` doesn't exist; wrap scalar: `full({1}, scalar_val, dtype, place)`

### Lua
- **All array indices are 1-based** — including string slicing: `arr["1:6"]` gives elements at 1-based positions 1..5 (stop exclusive). NOT `arr["0:5"]`.
- Properties vs methods: `arr.numel`, `arr.shape`, `arr.dtype` are **properties** (no parentheses). `arr:get(i)`, `arr:item()` are **methods**.
- `^` operator not overloaded for arrays — use `a * a` for square
- `//` not integer division — use `math.floor(a / b)`
- Default `Array()` params in C++ functions need `sol::optional<Array>` wrapper to avoid segfault
- String-based indexing: `arr["1:6"]` works, returns view
- `ins.signal.ca_cfar()` returns a pair-like table: result[1] = threshold, result[2] = detections

### Julia
- `set_device(Int64(0))` for CPU, `set_device(Int64(1))` for GPU. No `CPUPlace()/GPUPlace()`.
- `Insight.real_part(arr)` not `Insight.real(arr)`. Same for `imag_part`.
- `Insight.item(arr, 0)` — requires index arg (0-based), not `Insight.item(arr)`
- `Insight.mean(arr, axis=0, keepdims=true)` — uses keyword args
- `Insight.signal.get_window("hamming", n, fftbins=false)` — keyword arg `fftbins`
- **Missing operator overloads**: `+(::InsightArray, ::Number)`, `-(::InsightArray, ::Number)`, `/(::InsightArray, ::Number)` — must be added to `Insight.jl`
- `Insight.slice(arr, dim, start, stop)` where start/stop are 1-based and stop is exclusive. Binding must convert: `Int64(start-1), Int64(stop-1)` for 0-based C++.
- `from_data(arr)` creates on CPU regardless of `set_device`. Use `Insight.to(arr, 1)` to transfer to GPU.
- Signal module functions may be missing from export list; add to `Insight.jl` exports.

## Target Extraction Algorithm
Common pattern across all languages:
1. Transfer `energy` and `detections` to CPU
2. Scan 2D boolean `detections` array for `true` cells
3. Read energy values at those positions
4. Sort by energy descending, keep top N*10
5. Cluster by Euclidean distance (threshold ~3.0)
6. Deduplicate by range bin (keep strongest per range)
7. Return top-N targets

## Echo Simulation (Vectorized)
```python
# Python pattern — use 2D broadcasting
slow_times = arange(N_PULSES) * T_PRF
pulses = zeros([N_PULSES, N], complex128)
for delay, doppler in zip(delays, dopplers):
    delayed_1d = zeros([N], complex128)
    delayed_1d[ds:] = S_TX[0:N-ds]  # delayed copy
    phase = 2*PI*doppler*slow_times
    rot = to_complex(cos(phase), sin(phase))
    pulses = pulses + reshape(delayed_1d, [1, N]) * reshape(rot, [N_PULSES, 1])
pulses = pulses + to_complex(noise_r, noise_i)
```

## Adding Missing Bindings
When a function doesn't exist in a language binding:
1. **Julia**: Add C API wrapper in `insight_julia_capi.cpp`, then bind in `Insight.jl`
2. **Lua**: Add lambda with `sol::optional<>` in `insight_lua.cpp`
3. **Python**: Add `.def()` in `insight_py.cpp`
