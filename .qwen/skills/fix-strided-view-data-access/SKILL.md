---
name: fix-strided-view-data-access
description: "data<T>()[i] doesn't respect strides on sliced arrays — use at({i}).item<T>() for non-contiguous views"
source: auto-skill
extracted_at: '2026-05-29T20:42:45.731Z'
---

# Fix Strided View Data Access in Tests

## The Problem

`data<T>()` returns a raw pointer to the underlying memory buffer at the view's offset. Indexing it with `[i]` accesses the i-th element **in memory**, NOT the i-th element along the logical dimensions. For sliced arrays with step > 1 or negative step, this gives wrong results.

```cpp
Array a = arange(10, DType::F32, CPUPlace());  // [0, 1, 2, ..., 9]
Array d = a["1:8:2"];                           // logical: [1, 3, 5, 7], stride=2

// BUG: raw pointer access ignores stride
d.data<float>()[0]  // → 1.0 (correct by luck: offset matches)
d.data<float>()[1]  // → 2.0 (WRONG! should be 3.0 — reads next byte in memory)
d.data<float>()[2]  // → 3.0 (WRONG! should be 5.0)
```

This also affects:
- Reverse slices: `a["::-1"]` — stride is negative
- Multi-dim slices that change strides: `arr[":,1:-1"]` where dim1 stride != 1

## The Fix

Use `at({indices}).item<T>()` for element access on strided views:

```cpp
// CORRECT: at() respects strides
d.at({0}).item<float>()  // → 1.0 ✓
d.at({1}).item<float>()  // → 3.0 ✓
d.at({2}).item<float>()  // → 5.0 ✓
```

## When data<T>()[i] IS Safe

`data<T>()[i]` is safe when:
- The array is contiguous (stride = 1 for all dims)
- The array was created via `contiguous()`, `copy()`, or `to()`
- The array is a fresh `arange`, `zeros`, `ones`, `full`, `to_array` result
- You're accessing a row of a contiguous 2D array where the row stride happens to be 1

## How to Detect

In tests, search for `data<` access on sliced arrays:
```bash
grep -n "\.data<.*>\(\)\[" tests/
```

Cross-reference with any preceding `.slice()`, `operator[]`, or `.reshape()` that could make the array non-contiguous.

## Example from 2026-05-29

Added 6 string-slice tests (`StringSlice1D`, `StringSliceReverse`, etc.) that initially failed because they used `d.data<float>()[1]` on strided views. Fixed by switching to `d.at({i}).item<float>()`.

## Related Skills

- `fix-test-dangling-pointer` — different bug (temporary destruction), but same symptom (wrong data values in tests)
- Both bugs manifest as "test gets garbage values" — distinguish by checking if the array is a view (strided) vs a temporary (dangling)
