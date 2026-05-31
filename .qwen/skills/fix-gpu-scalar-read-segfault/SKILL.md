---
name: fix-gpu-scalar-read-segfault
description: Functions that read scalar values via .data<T>() or sum().data<T>() must compute on CPU to avoid GPU segfaults
source: auto-skill
extracted_at: '2026-05-30T02:00:00.000Z'
---

# Fix GPU Scalar Read Segfault

## Problem

When `set_device(GPUPlace(0))` is active (e.g., in CUDA tests), all composite ops
(`arange`, `full`, `ones`, `sin`, `cos`, `mul`, `add`, `sum`, etc.) create arrays on GPU.
Accessing `.data<T>()` on a GPU array from CPU code causes a segfault.

This affects any function that:
1. Calls `sum()`, `mean()`, `max()`, etc. and reads the scalar result
2. Calls `.data<double>()` to read individual elements
3. Uses `std::complex<double>*` reinterpret_cast on complex arrays

## Symptoms

- Segfault in CUDA tests on functions that work fine in CPU tests
- Crash at lines calling `.data<T>()` or `reinterpret_cast<...>(.data<char>())`
- Function works when called from CPU test but crashes from CUDA test

## Root Cause

Composite ops dispatch to the current device. When device is GPU:
```cpp
Array s = sum(h);           // sum runs on GPU, result is on GPU
double val = s.data<double>()[0];  // segfault: reading GPU memory from CPU
```

## Solution

Force CPU computation for functions that need scalar reads:

### Pattern 1: Explicit CPU Place for all creation ops

```cpp
Place cpu = CPUPlace();
Array m_arr = arange(0.0, double(n), 1.0, DType::F64, cpu);
Array result = ones({n}, DType::F64, cpu);
Array h = mul(result, full({n}, c, DType::F64, cpu));
// Now sum and .data<T>() are safe:
const double *hd = h.data<double>();
double total = 0.0;
for (int64_t i = 0; i < n; ++i) total += hd[i];
```

### Pattern 2: Transfer input to CPU first

```cpp
Place cpu = CPUPlace();
Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);
// Now all operations on x_cpu produce CPU arrays
Array Xf = fft::fft(x_cpu, n);
const std::complex<double> *data =
    reinterpret_cast<const std::complex<double> *>(Xf.data<char>());
```

### Pattern 3: Use raw pointer loops instead of composite ops for scalars

```cpp
// Instead of: Array s = sum(h); double val = s.data<double>()[0];
const double *hd = h.data<double>();
double val = 0.0;
for (int64_t i = 0; i < h.numel(); ++i) val += hd[i];
```

### Pattern 4: Transfer window/filter from get_window if needed

```cpp
Array win = get_window(window, numtaps, false);
if (win.place().kind() != DeviceKind::CPU) {
    win = win.to(cpu);
}
result = mul(result, win);  // CPU * CPU = CPU
```

## Affected Functions (examples from this session)

- `firwin` — scales by sum(h) or sum(h*(-1)^n)
- `firwin2` — scales by sum(h)
- `hilbert` — reads FFT output to apply frequency-domain filter
- `resample` — reads FFT output for frequency-domain manipulation
- `freq_shift` — reads cos/sin values to build complex exponential
- `cmplx_sort` — reads array data for sorting
- `lfilter_zi` — reads matrix elements for linear solve

## Key Rule

**If a function ever calls `.data<T>()` on its intermediate results, the entire
computation chain must produce CPU arrays. Use `CPUPlace()` explicitly on all
creation ops, or transfer inputs to CPU first.**

## How to Apply

When implementing a new signal function:
1. Check if it reads scalar values or array elements via `.data<T>()`
2. If yes, ensure ALL creation ops use `CPUPlace()`
3. If the function receives GPU input, transfer to CPU first
4. Transfer final result back to the original device if needed:
   ```cpp
   if (x.place().kind() != DeviceKind::CPU) result = result.to(x.place());
   ```
