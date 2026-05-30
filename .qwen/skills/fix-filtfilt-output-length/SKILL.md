---
name: fix-filtfilt-output-length
description: filtfilt must crop output to original signal length after each lfilter call since lfilter extends via full convolution
source: auto-skill
extracted_at: '2026-05-30T02:00:00.000Z'
---

# Fix filtfilt Output Length

## Problem

`filtfilt` (zero-phase filtering) calls `lfilter` twice (forward + backward).
`lfilter` for FIR filters uses full convolution, which extends the output:
- Input length: N
- Filter length: M
- lfilter output length: N + M - 1

After forward + backward filtering, the output is much longer than the input.

## Symptoms

- `filtfilt(b, a, x)` returns an array with `numel() > x.numel()`
- Tests comparing output length to input length fail
- Output values at the edges are incorrect (extra convolution artifacts)

## Solution

Crop the output to the original length after each `lfilter` call:

```cpp
Array filtfilt(const Array &b, const Array &a, const Array &x, int axis) {
    int ndim = x.shape().ndim();
    int ax = (axis < 0) ? axis + ndim : axis;

    // Forward filter
    Array y = lfilter(b, a, x, axis);

    // Crop to original length (lfilter extends via full convolution)
    int64_t orig_len = x.shape().dim(ax);
    if (y.shape().dim(ax) > orig_len) {
        int64_t nb = b.numel();
        int64_t start = nb - 1;  // skip the filter transient
        y = slice(y, {ax}, {start}, {start + orig_len});
    }

    // Reverse along axis, filter again, reverse back
    Array y_rev = flip(y, ax);
    y_rev = lfilter(b, a, y_rev, ax);

    // Crop again
    if (y_rev.shape().dim(ax) > orig_len) {
        int64_t nb = b.numel();
        int64_t start = nb - 1;
        y_rev = slice(y_rev, {ax}, {start}, {start + orig_len});
    }

    return flip(y_rev, ax);
}
```

## Key Points

1. The crop offset is `nb - 1` where `nb = b.numel()` (filter length)
2. Crop AFTER each `lfilter` call, not just at the end
3. For IIR filters (a has length > 1), lfilter may not extend the output,
   but the crop logic still works (no-op if output length == input length)
4. The `slice` function uses `{ax}, {start}, {end}` syntax for axis-wise slicing

## How to Apply

Any function that chains multiple `lfilter` or `convolve` calls must crop
intermediate results to avoid length accumulation. This applies to:
- `filtfilt` (forward-backward filtering)
- `sosfiltfilt` (SOS forward-backward)
- Any cascaded filtering where output length should match input
