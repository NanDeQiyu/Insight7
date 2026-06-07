---
name: fix-full-kernel-view-offset
description: "full kernel ignores out->offset, writing to parent array start instead of view position — affects fill_() on views"
source: auto-skill
---

# Fix full kernel for view offset

## The Problem

The `full` kernel (`backends/cpu/kernels/creation/full.cpp` and `.cu`) writes to `out->data` directly, ignoring `out->offset`. When `out` is a view (e.g., `arr["1,1"]`), the data pointer points to the parent array's start, and `offset` indicates where the view begins. Without adding offset, the kernel writes to position 0 instead of the view's actual position.

```cpp
// WRONG: ignores offset
float *dst = static_cast<float *>(out->data);
dst[i] = val;

// CORRECT: applies offset
float *dst = static_cast<float *>(out->data) + out->offset;
dst[i] = val;
```

## Symptom

```python
a = ins.from_numpy(np.arange(9).reshape(3,3))
a["1,1"] = 99  # Should write to position [1,1] (offset 4)
# Bug: writes to position [0,0] (offset 0)
print(a.numpy())  # [[99, 1, 2], [3, 4, 5], [6, 7, 8]] ← WRONG
```

Scalar view `a["1,1"]` has `is_contiguous() == true` (0-dim), so `fill_()` dispatches directly to the `full` kernel. The kernel writes to `out->data[0]` which is the parent's position 0.

## The Fix

### CPU kernel (`backends/cpu/kernels/creation/full.cpp`)

Add offset to the data pointer in every dtype case:

```cpp
int64_t offset = out->offset;
// ...
case INSIGHT_DTYPE_F64: {
    double *dst = static_cast<double *>(out->data) + offset;
    for (int64_t i = 0; i < n; ++i)
        dst[i] = fill_val;
    break;
}
```

### CUDA kernel (`backends/cuda/kernels/creation/full.cu`)

Compute offset-adjusted pointer before the switch:

```cpp
size_t elem_size = insight_dtype_size(dtype);
char *data_with_offset = static_cast<char *>(out->data) + out->offset * elem_size;
// Then cast data_with_offset in each case:
full_kernel<<<blocks, threads>>>(reinterpret_cast<float *>(data_with_offset), val, n);
```

## How to detect

Test scalar element assignment on a 2D array:
```python
a = ins.from_numpy(np.arange(9, dtype=np.float64).reshape(3,3))
a["1,1"] = 99
assert a.numpy()[1,1] == 99  # Fails without fix (writes to [0,0])
```

## Root cause

The `full` kernel was designed for creating new arrays (where offset is always 0), not for filling views. When `fill_()` reuses the `full` kernel for contiguous views, the offset must be respected.

## Related Skills

- `add-array-setitem` — fill_() dispatches to full kernel for contiguous views
- `fix-contiguous-copy-noncontiguous-output` — similar pattern (kernel ignores output metadata)
- `fix-strided-view-data-access` — general category of view offset bugs
