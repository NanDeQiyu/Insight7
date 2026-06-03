---
name: fix-scatter-broadcast-shape
description: scatter() source array must be 1D — reshaping to [N,1] causes broadcast_to error
source: auto-skill
extracted_at: '2026-06-01T14:22:00.581Z'
---

# Fix scatter() broadcast shape error

## Symptom

```
broadcast_to: cannot broadcast shape [N, 1] to [N]
  [/path/to/broadcast.cpp:85]
```

Occurs when calling `scatter(tensor, dim, index, source)` where `source` has been reshaped from 1D `[N]` to 2D `[N, 1]`.

## Root Cause

Insight7's `scatter` implementation broadcasts the `source` tensor to match the index shape at the scattered dimension. When source is `[N, 1]` and the expected shape at that dimension is `[N]` (1D), the broadcast fails because it cannot collapse a 2D shape to 1D.

This commonly happens in spectrogram/stft code where a column is extracted and scattered into a 2D accumulator:

```cpp
// BUG: reshape makes source 2D
Array col = slice(Xf_real, 0, 0, static_cast<int>(freq_len));
Sxx_2d = scatter(Sxx_2d, 1, col_idx, reshape(col, {freq_len, 1}));  // FAILS
```

## Fix

Pass the source as 1D directly — no reshape needed:

```cpp
// FIXED: col is already [freq_len] (1D)
Array col = slice(Xf_real, 0, 0, static_cast<int>(freq_len));
Sxx_2d = scatter(Sxx_2d, 1, col_idx, col);  // OK
```

## Verification

```python
import insight as ins
import numpy as np

# Should work:
a = ins.zeros([129, 15], ins.float64)
idx = ins.from_numpy(np.array([0], dtype=np.int64))
src_1d = ins.ones([3], ins.float64)
r = ins.scatter(a, 1, idx, src_1d)  # OK

# Should fail:
src_2d = ins.ones([3, 1], ins.float64)
r = ins.scatter(a, 1, idx, src_2d)  # broadcast_to error
```

## General Rule

Always pass 1D source arrays to `scatter()`. If the source comes from a slice that might have extra dimensions, use `squeeze()` instead of `reshape()`.
