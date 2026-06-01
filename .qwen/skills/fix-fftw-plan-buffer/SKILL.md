---
name: fix-fftw-plan-buffer
description: FFTW C2C plan must use actual execution buffer, not dummy buffers (n>=64 gives wrong results)
source: auto-skill
extracted_at: '2026-05-31T04:00:00.000Z'
---

# FFTW Plan Buffer Mismatch Bug

## Problem

FFTW `fftw_execute_dft(plan, real_buf, real_buf)` gives **wrong results for n>=64** when the plan was created with different dummy buffers.

**Symptoms:**
- FFT of impulse `[1,0,0,...,0]` (n=128) returns `[34.2+8.8i, ...]` instead of `[1+0i, ...]`
- n<=32 works correctly, n>=64 is wrong
- Affects both F32 and F64
- Only affects C2C FFT (R2C/C2R work fine with dummy buffers)

## Root Cause

FFTW's `FFTW_ESTIMATE` mode still depends on buffer addresses for alignment hints. When `fftw_plan_many_dft` is called with dummy buffers, the plan encodes assumptions about memory layout. `fftw_execute_dft` with different buffers violates these assumptions for larger transforms.

## Fix

In `backends/cpu/kernels/fft/fft_c2c.cpp`, create the plan with the actual execution buffer:

```cpp
// WRONG: plan with dummy, execute with real buffer
fftw_plan plan = fft_ensure_plan_f64(fft_len, batch, direction, FFT_KIND_C2C);
fftw_execute_dft(plan, (fftw_complex*)data, (fftw_complex*)data);

// CORRECT: create plan with actual buffer, execute, destroy
int n = (int)fft_len;
fftw_plan plan = fftw_plan_many_dft(1, &n, batch,
    (fftw_complex*)data, NULL, 1, n,
    (fftw_complex*)data, NULL, 1, n,
    direction, FFTW_ESTIMATE);
fftw_execute(plan);
fftw_destroy_plan(plan);
```

## Impact

- Fixes all C2C FFT operations for n>=64
- Enables correct complex fftconvolve via FFT path (no more O(n*m) workaround)
- Fixes radar Doppler processing (128-point FFT)

## Caching Strategy

The final solution uses a **split caching strategy**:

- **C2C transforms**: Plan created with actual buffer each time (correct but slow). Cache by buffer address — if same buffer is reused, return cached plan.
- **R2C/C2R transforms**: Plan created with dummy buffers and cached by (n, batch, direction, kind). `fftw_execute_dft_r2c/c2r` works correctly with different buffers.

```cpp
// In common_impl.cpp:
if (kind == FFT_KIND_C2C) {
    // Always create with actual buffer
    c->plan_f64 = fftw_plan_many_dft(1, &n_int, batch,
        (fftw_complex*)buf, NULL, 1, n_int,
        (fftw_complex*)buf, NULL, 1, n_int,
        direction, FFTW_ESTIMATE);
} else {
    // R2C/C2R: dummy buffers work fine
    fftw_complex *di = fftw_malloc(n * sizeof(fftw_complex));
    fftw_complex *dout = fftw_malloc(n * sizeof(fftw_complex));
    // ... create plan with di/dout ...
    fftw_free(di); fftw_free(dout);
}
```

## What DOESN'T Work

1. **FFTW_ESTIMATE + dummy buffers + fftw_execute_dft**: Wrong for n>=64
2. **FFTW_PATIENT + dummy buffers + fftw_execute_dft**: Correct but causes heap corruption (malloc_consolidate error)
3. **FFTW_MEASURE + dummy buffers**: Same heap corruption issue

## Performance Impact

- C2C FFT: ~1.4s per call (plan recreation each time)
- R2C/C2R FFT: ~0.04s per call (cached plan)
- fftconvolve with complex inputs: slow due to multiple C2C FFT calls
- For hot paths, consider pre-allocated FFT buffers to enable plan reuse

## Verification

```python
import insight as ins; import numpy as np; ins.init(['cpu'])
for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    x = np.zeros(n, dtype=np.complex128); x[0] = 1
    r = ins.fft(ins.from_numpy(x), n).numpy()
    assert abs(r[0] - 1.0) < 0.01, f"FFT broken at n={n}"
```

## Files Modified

- `backends/cpu/kernels/fft/fft_c2c.cpp` — C2C kernel
- `src/ops/signal/convolution.cpp` — fftconvolve complex path restored
