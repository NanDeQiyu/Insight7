---
name: bind-signal-subnamespace
description: Binding ins::signal:: sub-namespace functions to Python/Lua/Julia with proper type conversion
source: auto-skill
extracted_at: '2026-05-30T20:00:00.010Z'
updated: '2026-06-01'
---

# Binding ins::signal:: Sub-namespace to Scripting Languages

## Problem

The `ins::signal::` namespace contains ~89 functions across 14 sub-modules
(windows, waveforms, bsplines, filter_design, convolution, filtering,
spectral_analysis, wavelets, acoustics, peak_finding, demod, estimation,
radartools, io). All sub-modules have CPU+CUDA backend kernels with the
`signal_` prefix convention. dtype support: CPU (F64, F32), CUDA (F64, F32, F16, BF16).

## Key Challenges

### 1. Nested namespace
Functions are in `ins::signal::welch`, not `ins::welch`. Bindings must
expose them as `ins.signal.welch` (Python), `ins.signal.welch` (Lua),
or `Insight.signal_welch` (Julia).

### 2. Return types
Some functions return structs (`SpectralResult`, `SpectrogramResult`,
`GaussPulseResult`) or pairs (`std::pair<double, double>`).
These need per-language handling:
- **Python**: Return as tuple or named tuple
- **Lua**: Return as table `{f=..., Pxx=...}`
- **Julia**: Return as NamedTuple or multiple return values

### 3. String parameters
Functions like `firwin(numtaps, cutoff, window="hamming", pass_zero="lowpass")`
use `std::string` and `std::vector<double>`. Each language needs conversion:
- **Python**: pybind11 handles `str` → `std::string` automatically
- **Lua**: sol2 handles `string` → `std::string` automatically
- **Julia**: C ABI needs `Cstring` → `const char*` conversion

### 4. Function overloads
`get_window` has 2 overloads (with/without param). pybind11 handles this
naturally. sol2 needs explicit overload resolution. Julia C ABI needs
separate C functions for each overload.

## Solution Pattern

### Python (pybind11)

```cpp
// In bindings/python/insight_py.cpp
auto signal_m = m.def_submodule("signal", "Signal processing functions");

// Windows
signal_m.def("hann", [](int64_t M, bool sym) {
    return ins::signal::hann(M, sym);
}, py::arg("M"), py::arg("sym") = true);
signal_m.def("hamming", [](int64_t M, bool sym) {
    return ins::signal::hamming(M, sym);
}, py::arg("M"), py::arg("sym") = true);
// ... all 24 window functions

// Spectral analysis - return as tuple
signal_m.def("welch", [](const ins::Array &x, double fs, const std::string &window,
                          int64_t nperseg, int64_t noverlap) {
    auto result = ins::signal::welch(x, fs, window, nperseg, noverlap);
    return py::make_tuple(result.f, result.Pxx);
}, py::arg("x"), py::arg("fs") = 1.0, py::arg("window") = "hann",
   py::arg("nperseg") = 256, py::arg("noverlap") = 0);
```

### Lua (sol2)

```cpp
// In bindings/lua/insight_lua.cpp
auto signal = lua.create_named_table("signal");

signal.set_function("hann", [](int64_t M, bool sym) {
    return ins::signal::hann(M, sym);
});
signal.set_function("welch", [](const ins::Array &x, double fs,
                                 const std::string &window, int64_t nperseg) {
    auto result = ins::signal::welch(x, fs, window, nperseg);
    // Return as table
    auto tbl = lua.create_table();
    tbl["f"] = result.f;
    tbl["Pxx"] = result.Pxx;
    return tbl;
});
```

### Julia (C ABI)

```cpp
// In bindings/julia/insight_julia_capi.cpp
extern "C" {
INSIGHT_JL_API void *insight_jl_signal_welch(
    void *x, double fs, const char *window, int64_t nperseg, int64_t noverlap,
    void **out_f, void **out_pxx) {
    auto *arr = reinterpret_cast<ins::Array *>(x);
    auto result = ins::signal::welch(*arr, fs, window, nperseg, noverlap);
    *out_f = new ins::Array(result.f);
    *out_pxx = new ins::Array(result.Pxx);
    return nullptr; // success
}
}
```

```julia
# In bindings/julia/Insight.jl
function signal_welch(x::Array, fs::Float64=1.0, window::String="hann",
                      nperseg::Int64=256, noverlap::Int64=0)
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    pxx_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_signal_welch, libinsight_julia), Ptr{Cvoid},
          (Ptr{Cvoid}, Float64, Cstring, Int64, Int64, Ref{Ptr{Cvoid}}, Ref{Ptr{Cvoid}}),
          x.ptr, fs, window, nperseg, noverlap, f_ref, pxx_ref)
    return (Array(f_ref[]), Array(pxx_ref[]))
end
```

## Sub-module mapping

| C++ module | Python | Lua | Julia | Backend Kernels |
|-----------|--------|-----|-------|-----------------|
| `ins::signal::hann` | `ins.signal.hann` | `ins.signal.hann` | `signal_hann` | 12 windows kernels |
| `ins::signal::welch` | `ins.signal.welch` | `ins.signal.welch` | `signal_welch` | 3 spectral kernels |
| `ins::signal::firwin` | `ins.signal.firwin` | `ins.signal.firwin` | `signal_firwin` | 1 filter_design kernel |
| `ins::signal::lfilter` | `ins.signal.lfilter` | `ins.signal.lfilter` | `signal_lfilter` | 8 filtering kernels |
| `ins::signal::convolve` | `ins.signal.convolve` | `ins.signal.convolve` | `signal_convolve` | 3 convolution kernels |
| `ins::signal::morlet` | `ins.signal.morlet` | `ins.signal.morlet` | `signal_morlet` | 3 wavelet kernels |
| `ins::signal::sawtooth` | `ins.signal.sawtooth` | `ins.signal.sawtooth` | `signal_sawtooth` | 5 waveform kernels |
| `ins::signal::cubic` | `ins.signal.cubic` | `ins.signal.cubic` | `signal_cubic` | 3 bspline kernels |
| `ins::signal::mel2hz` | `ins.signal.mel2hz` | `ins.signal.mel2hz` | `signal_mel2hz` | 5 acoustics kernels |
| `ins::signal::argrelmax` | `ins.signal.argrelmax` | `ins.signal.argrelmax` | `signal_argrelmax` | 2 peak_finding kernels |
| `ins::signal::fm_demod` | `ins.signal.fm_demod` | `ins.signal.fm_demod` | `signal_fm_demod` | 0 (pure composite) |
| `ins::signal::KalmanFilter` | `ins.signal.KalmanFilter` | `ins.signal.KalmanFilter` | `KalmanFilter` | 1 estimation kernel |
| `ins::signal::ca_cfar` | `ins.signal.ca_cfar` | `ins.signal.ca_cfar` | `signal_ca_cfar` | 2 radar kernels |
| `ins::signal::read_bin` | `ins.signal.read_bin` | `ins.signal.read_bin` | `signal_read_bin` | 2 io kernels |

## Struct return handling

For `SpectralResult { Array f; Array Pxx; }`:
- Python: `py::make_tuple(result.f, result.Pxx)` — natural tuple unpacking
- Lua: `sol::table` with named keys — `result.f`, `result.Pxx`
- Julia: Multiple return values via C ABI output pointers

For `std::pair<double, double>` (e.g., vectorstrength):
- Python: `py::make_pair(strength, phase)`
- Lua: `sol::table` or just return two values
- Julia: Two `Ref{Float64}` output parameters

## Testing pattern

Each language needs:
1. **Smoke test** — function exists and returns defined array
2. **Numerical test** — compare output against scipy reference values
3. **CPU test** — on CPUPlace
4. **CUDA test** — on GPUPlace, transfer back to CPU for verification

```python
# Python numerical test
def test_welch_peak():
    x = np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
    f, Pxx = ins.signal.welch(ins.from_numpy(x), 256.0, "hann", 256, 128)
    f_np = f.numpy()
    Pxx_np = Pxx.numpy()
    peak_idx = np.argmax(Pxx_np[1:]) + 1
    assert abs(f_np[peak_idx] - 10.0) < 2.0
```

## Verification

```bash
# Python
cd build && python -m pytest tests/bindings/test_signal_binding.py -v

# Lua (busted)
busted tests/bindings/test_signal_binding.lua

# Julia
julia tests/bindings/test_signal_binding.jl
```
