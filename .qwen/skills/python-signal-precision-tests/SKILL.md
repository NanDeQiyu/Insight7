---
name: python-signal-precision-tests
description: Writing Python precision alignment tests for Insight7 signal functions vs scipy.signal
source: auto-skill
extracted_at: '2026-05-31T01:00:00.000Z'
---

# Python Signal Precision Alignment Tests vs scipy.signal

## Purpose

Verify that Insight7's signal processing functions produce numerically correct results by comparing against scipy.signal reference implementations.

## File Location

`tests/bindings/test_signal_precision.py`

## Setup Pattern

```python
import numpy as np
import pytest

try:
    import insight as ins
    ins.init(["cpu"])  # NOTE: must be a list, not a string
    from scipy import signal as scipy_signal
except ImportError:
    pytest.skip("scipy or insight not available", allow_module_level=True)

def to_insight(arr, dtype=np.float64):
    return ins.from_numpy(arr.astype(np.float64))

def to_numpy(arr):
    return arr.numpy()  # InsightArray has .numpy() method
```

## Key Convention: Window Symmetry

Insight7's `sym=True` (default) corresponds to scipy's `fftbins=False` (NOT the default).
When comparing windows, always use `fftbins=False` in scipy:

```python
scipy_w = scipy_signal.get_window("hann", 32, fftbins=False)
ins_w = ins.signal.hann(32).numpy()
np.testing.assert_allclose(ins_w, scipy_w, atol=1e-10)
```

## Test Categories

### 1. Exact Numerical Match (tight tolerance)
For functions where implementations should match exactly:
- Windows (with correct fftbins convention)
- kaiser_beta, kaiser_atten
- argrelmax, argrelmin
- unwrap, sinc, convolve (top-level)

```python
np.testing.assert_allclose(ins_out, scipy_out, atol=1e-10)
```

### 2. Structural Correctness (relaxed)
For functions where implementations may differ in normalization or padding:
- firwin (different algorithm: windowed-sinc vs frequency sampling)
- fftconvolve (may use different FFT normalization)
- lfilter (may return full convolution length vs same length)
- welch/periodogram (different FFT size or window application)

```python
assert len(ins_out) == len(scipy_out), "Output length mismatch"
assert np.all(np.isfinite(ins_out)), "Output should be finite"
# For PSD: assert np.all(ins_pxx >= 0), "PSD should be non-negative"
# For lowpass: dc_gain = np.sum(ins_b); assert abs(dc_gain - 1.0) < 0.1
```

### 3. Skipped Tests
For scipy APIs not available in the installed version:
```python
@pytest.mark.skip(reason="scipy.signal.ricker not available in this version")
def test_ricker(self):
    pass
```

## Known Differences (do NOT assert equality)

| Function | Insight7 Behavior | scipy Behavior |
|----------|-------------------|----------------|
| `firwin` | Windowed-sinc method | Frequency sampling |
| `lfilter` (FIR) | Returns `len(x) + len(b) - 1` | Returns `len(x)` |
| `welch`/`periodogram` | Uses `nperseg/2+1` FFT bins | Uses `nfft/2+1` FFT bins |
| `bartlett` window | Slightly different endpoint values | Standard bartlett formula |

## Running

```bash
cd ~/plum/Insight7
PYTHONPATH=bindings/python LD_LIBRARY_PATH=build/backends/cpu \
  python3 -m pytest tests/bindings/test_signal_precision.py -v
```
